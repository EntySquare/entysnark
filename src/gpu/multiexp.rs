use super::error::{GPUError, GPUResult};
use super::locks;
use super::sources;
use super::utils;
use crate::bls::Engine;
use crate::multicore::Worker;
use crate::multiexp::{multiexp as cpu_multiexp, FullDensity};
use ff::{PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::{error, info};
use rayon::prelude::*;
use rust_gpu_tools::*;
use std::any::TypeId;
use std::sync::Arc;

use std::sync::mpsc;
extern crate scoped_threadpool;
use scoped_threadpool::Pool;
use std::time::Instant;


// const MAX_WINDOW_SIZE: usize = 11; // 10;
const LOCAL_WORK_SIZE: usize = 256;
// const MEMORY_PADDING: f64 = 0.1f64; // 0.2f64; // Let 20% of GPU memory be free

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                Ok(0f64)
            }
        })
        .unwrap_or(0f64)
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
where
    E: Engine,
{
    program: opencl::Program,

    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

// fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
//     // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
//     8 * core_count / num_windows
// }


// fn calc_window_size(n: usize, exp_bits: usize, core_count: usize) -> usize {
//     // window_size = ln(n / num_groups)
//     // num_windows = exp_bits / window_size
//     // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
//     // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
//     // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
//     //
//     // Thus we need to solve the following equation:
//     // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
//     let lower_bound = (((exp_bits * n) as f64) / ((2 * core_count) as f64)).ln();
//     for w in 0..MAX_WINDOW_SIZE {
//         if (w as f64) + (w as f64).ln() > lower_bound {
//             return w;
//         }
//     }

//     MAX_WINDOW_SIZE
// }
// fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize) -> usize {
//     // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
//     // n = e^window_size * window_size * 2 * core_count / exp_bits
//     (((max_window_size as f64).exp() as f64)
//         * (max_window_size as f64)
//         * 2f64
//         * (core_count as f64)
//         / (exp_bits as f64))
//         .ceil() as usize
// }
// fn calc_chunk_size<E>(mem: u64, core_count: usize) -> usize
// where
//     E: Engine,
// {
//     let aff_size = std::mem::size_of::<E::G1Affine>() + std::mem::size_of::<E::G2Affine>();
//     let exp_size = exp_size::<E>();
//     let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();
//     ((((mem as f64) * (1f64 - MEMORY_PADDING)) as usize)
//         - (2 * core_count * ((1 << MAX_WINDOW_SIZE) + 1) * proj_size))
//         / (aff_size + exp_size)
// }

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(d: opencl::Device, priority: bool) -> GPUResult<SingleMultiexpKernel<E>> {
        let src = sources::kernel::<E>(d.brand() == opencl::Brand::Nvidia);

        // let exp_bits = exp_size::<E>() * 8;
        //let core_count = utils::get_core_count(&d);
        let core_count = 8704;

        // let mem = d.memory();
        // let max_n = calc_chunk_size::<E>(mem, core_count);
        // let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        // let n = std::cmp::min(max_n, best_n);
        let n = 568; // Not used

        Ok(SingleMultiexpKernel {
            program: opencl::Program::from_opencl(d, &src)?,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
        set_window_size: usize,
        bus_id: u32,
        times: u32,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let exp_bits = exp_size::<E>() * 8;
        // let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let window_size = set_window_size;
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        // let num_groups = calc_num_groups(self.core_count, num_windows);
        let num_groups =  2 * self.core_count / num_windows;
        let bucket_len = 1 << window_size;
        // println!("[{} - {}] SingleMultiexpKernel.multiexp:  exp_bits:{},window_size:{},num_windows:{},num_groups:{},bucket_len:{}",bus_id, times, exp_bits,window_size,num_windows,num_groups,bucket_len);

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.
        // let size1 = std::mem::size_of::<G>();
        // let size2 = std::mem::size_of::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>();
        // let size3 = std::mem::size_of::<<G as CurveAffine>::Projective>();
        // let mem1 = size1 * n;
        // let mem2 = size2 * n;
        // //2 * self.core_count =`num_groups` * `num_windows`
        // let mem3 = size3 * num_groups * num_windows * bucket_len;
        // let mem4 = size3 * num_groups * num_windows ;
        // println!("[{} - {}] SingleMultiexpKernel.multiexp: CurveAffine size1:{} ,PrimeField size2:{} ,Projective size3:{} ,mem1:{} ,mem2:{} ,mem3:{} ,mem4:{} ,GPU mem need: {}Mbyte",bus_id, times, size1,size2,size3,mem1,mem2,mem3,mem4,(mem1 + mem2 + mem3 + mem4) / (1024 * 1024));

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.
        let mut base_buffer = self.program.create_buffer::<G>(n)?;
        base_buffer.write_from(0, bases)?;
        let mut exp_buffer = self
            .program
            .create_buffer::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>(n)?;
        exp_buffer.write_from(0, exps)?;

        let bucket_buffer = self
            .program
            .create_buffer::<<G as CurveAffine>::Projective>(num_groups * num_windows  * bucket_len)?;
        let result_buffer = self
            .program
            .create_buffer::<<G as CurveAffine>::Projective>(num_groups * num_windows )?;
        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut global_work_size = num_windows * num_groups;
        global_work_size +=
            (LOCAL_WORK_SIZE - (global_work_size % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        // println!("[{} - {}] SingleMultiexpKernel.multiexp: global_work_size:{},num_windows:{},num_groups:{},LOCAL_WORK_SIZE:{}",bus_id, times, global_work_size,num_windows,num_groups,LOCAL_WORK_SIZE);

        let kernel = self.program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            },
            global_work_size,
            None,
        );

        kernel
            .arg(&base_buffer)
            .arg(&bucket_buffer)
            .arg(&result_buffer)
            .arg(&exp_buffer)
            .arg(n as u32)
            .arg(num_groups as u32)
            .arg(num_windows as u32)
            .arg(window_size as u32)
            .run()?;

        let mut results = vec![<G as CurveAffine>::Projective::zero(); num_groups * num_windows];
        result_buffer.read_into(0, &mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }
        Ok(acc)
    }
}

// A struct that containts several multiexp kernels for different devices
pub struct MultiexpKernel<E>
where
    E: Engine,
{
    kernels: Vec<SingleMultiexpKernel<E>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(priority: bool) -> GPUResult<MultiexpKernel<E>> {
        let lock = locks::GPULock::lock_all();

        let devices = opencl::Device::all();
        let mut index = 0;

        let kernels: Vec<_> = devices
            .into_iter()
            .map(|d| (d, SingleMultiexpKernel::<E>::create(d.clone(), priority)))
            .filter_map(|(device, res)| {
                if let Err(ref e) = res {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                // 只用一个GPU
                // if index == 1{
                //     index += 1;
                //     res.ok()
                // }else{
                //     index += 1;
                //     None
                // }
                // 多个GPU
                res.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels.len(),
            get_cpu_utilization()
        );
        for (_, k) in kernels.iter().enumerate() {
            info!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                k.program.device().bus_id().unwrap(), // i, // Modified by long 20210312
                k.program.device().name(),
                k.n
            );
        }
        Ok(MultiexpKernel::<E> {
            kernels,
            _lock: lock,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
        <G as groupy::CurveAffine>::Engine: crate::bls::Engine,
    {

        let num_devices = self.kernels.len();
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + n)];
        let exps = &exps[..n];

        let cpu_n = ((n as f64) * get_cpu_utilization()) as usize;
        let n = n - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);
        println!("main MultiexpKernel.multiexp:cpu_utilization:{} , cpu_n={}, gpu_n={}",get_cpu_utilization(),cpu_n,n);

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;
        // println!("main MultiexpKernel.multiexp: exp_num:{} , num_devices:{} , chunk_size:{}",n,num_devices, chunk_size);

        let mut acc = <G as CurveAffine>::Projective::zero();

        // let results = crate::multicore::THREAD_POOL.install(|| {
        //     if n > 0 {
        // concurrent computing
        let (tx_gpu, rx_gpu) = mpsc::channel();
        let (tx_cpu, rx_cpu) = mpsc::channel();
        let mut scoped_pool = Pool::new(2);
        scoped_pool.scoped(|scoped| {
            // GPU
            scoped.execute(move || {
                let start = Instant::now();
                println!("main MultiexpKernel.multiexp: ================================ gpu multiexp start ================================");
                let results = if n > 0 {
                   // println!("MultiexpKernel.multiexp: \n total bases.len():{},\n exps.len():{},\n chunk_size:{}",bases.len(),exps.len(),chunk_size);
                    bases
                        .par_chunks(chunk_size)
                        .zip(exps.par_chunks(chunk_size))
                        .zip(self.kernels.par_iter_mut())
                        .map(|((bases, exps), kern)| -> Result<<G as CurveAffine>::Projective, GPUError> {
                            let bus_id = kern.program.device().bus_id().unwrap();
                            // println!(
                            //     "[{}] Multiexp: Device {}: {} core count:{} (Chunk-size: {})",
                            //     bus_id,
                            //     bus_id, // i, // Modified by long 20210312
                            //     kern.program.device().name(),
                            //     kern.core_count,
                            //     kern.n
                            // );
                           // println!("MultiexpKernel.multiexp: \n par_chunks bases.len():{},\n exps.len():{},\n chunk_size:{}",bases.len(),exps.len(),chunk_size);
                            let mut acc = <G as CurveAffine>::Projective::zero();
                            // let single_chunk_size = 33554466; //理论最佳 2台gpu 134217727/4 = 33554431.75  33554466  1台gpu 134217727/3=44739242.333333336 44739288
                            //let single_chunk_size = (((chunk_size as f64) / (4 as f64)).ceil() + 34 as f64 ) as usize;
                            let single_chunk_size = (33554466 as f64 *(1 as f64 - get_cpu_utilization()) as f64).ceil() as usize;
                            let mut set_window_size = 11; //grouprate=>window_size : 2=>11,4=>11,8=>10,16=>9
                            let size_result = std::mem::size_of::<<G as CurveAffine>::Projective>();
                            // println!("GABEDEBUG: start size_result:{}", size_result);
                            if size_result > 144 {
                                // single_chunk_size = 37282740; //1台gpu时设置
                                set_window_size = 8; //grouprate=>window_size : 2=>8,4=>8,8=>8,16=>7
                            }
                            // println!("[{}] MultiexpKernel.multiexp:  chunks bases.len():{} , exps.len():{} , chunk_size:{}", bus_id,bases.len(), exps.len(), single_chunk_size);
                            let mut times :u32 = 1;
                            for (bases, exps) in bases.chunks(single_chunk_size).zip(exps.chunks(single_chunk_size)) {
                                let now = Instant::now();
                                // println!("[{} - {}] MultiexpKernel.multiexp: ===========> Single multiexp start <=========== ",bus_id,times);
                                let result = kern.multiexp(bases, exps, bases.len(), set_window_size,bus_id,times)?;
                                // println!("[{} - {}] MultiexpKernel.multiexp: ===========> Single multiexp cost:{:?} <=========== ",bus_id,times,now.elapsed());
                                times += 1;
                                acc.add_assign(&result);
                            }
                            Ok(acc)
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                println!("main MultiexpKernel.multiexp: ================================ gpu multiexp cost:{:?} end ================================",start.elapsed());
                tx_gpu.send(results).unwrap();

            });
            // CPU
            scoped.execute(move || {
                let start = Instant::now();
                println!("main MultiexpKernel.multiexp: ================================ cpu multiexp start ================================ ");
                let cpu_acc = cpu_multiexp(
                    &pool,
                    (Arc::new(cpu_bases.to_vec()), 0),
                    FullDensity,
                    Arc::new(cpu_exps.to_vec()),
                    &mut None,
                );
                println!("main MultiexpKernel.multiexp: ================================ cpu multiexp cost:{:?} end ================================ ",start.elapsed());
                let cpu_r = cpu_acc.wait().unwrap();

                tx_cpu.send(cpu_r).unwrap();
            });
        });

        // waiting results...
        let results = rx_gpu.recv().unwrap();
        let cpu_r = rx_cpu.recv().unwrap();

        for r in results {
            match r {
                Ok(r) => acc.add_assign(&r),
                Err(e) => return Err(e),
            }
        }

        // acc.add_assign(&cpu_acc.wait().unwrap());
        acc.add_assign(&cpu_r);

        Ok(acc)
    }
}
