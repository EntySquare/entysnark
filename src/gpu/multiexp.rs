extern crate scoped_threadpool;

use std::any::TypeId;
use std::ops::AddAssign;
use std::sync::{Arc};
use std::sync::mpsc;
use std::time::Instant;

use ff::PrimeField;
use group::{Group, prime::PrimeCurveAffine};
use log::{debug, error, info};
use pairing::Engine;
use rayon::prelude::*;
use rust_gpu_tools::{Device, Program, program_closures};
use scoped_threadpool::Pool;

use crate::multicore::Worker;
use crate::multiexp::{FullDensity, multiexp as cpu_multiexp};

use super::{GpuEngine, locks, program};
use super::error::{GPUError, GPUResult};

// const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
// const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .map_or(0f64, |v| match v.parse() {
            Ok(val) => val,
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                0f64
            }
        })
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
    where
        E: Engine + GpuEngine,
{
    program: Program,
    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

// fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
//     // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
//     2 * core_count / num_windows
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
//
//     MAX_WINDOW_SIZE
// }
//
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
//
// fn calc_chunk_size<E>(mem: u64, core_count: usize) -> usize
//     where
//         E: Engine,
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
        E: Engine + GpuEngine,
{
    pub fn create(device: &Device, priority: bool) -> GPUResult<SingleMultiexpKernel<E>> {
        //let exp_bits = exp_size::<E>() * 8;
        //let core_count = utils::get_core_count(&device.name());
        //info!("core_count: {}", core_count);
        let core_count = 8704;
        // let mem = device.memory();
        // let max_n = calc_chunk_size::<E>(mem, core_count);
        // let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        // let n = std::cmp::min(max_n, best_n);
        // let n = 33554466;
        let n = 16777216;
        let program = program::program::<E>(device)?;

        Ok(SingleMultiexpKernel {
            program,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<G::Scalar as PrimeField>::Repr],
        n: usize,
        set_window_size: usize,
    ) -> GPUResult<<G as PrimeCurveAffine>::Curve>
        where
            G: PrimeCurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let exp_bits = exp_size::<E>() * 8;
        //let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let window_size = set_window_size;
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        // let num_groups = calc_num_groups(self.core_count, num_windows);
        let num_groups = 2 * self.core_count / num_windows;
        let bucket_len = 1 << window_size;

        let size1 = std::mem::size_of::<G>();
        let size2 = std::mem::size_of::<<G::Scalar as PrimeField>::Repr>();
        let size3 = std::mem::size_of::<<G as PrimeCurveAffine>::Curve>();
        let mem1 = size1 * n;
        let mem2 = size2 * n;
        //2 * self.core_count =`num_groups` * `num_windows`
        let mem3 = size3 * 2 * self.core_count * bucket_len;
        let mem4 = size3 * 2 * self.core_count;
        debug!("bases.len() : {}, n:{}, CurveAffine size1:{} ,PrimeField size2:{} ,Projective size3:{}", bases.len(), n, size1, size2, size3);
        debug!("mem1:{} ,mem2:{} ,mem3:{} ,mem4:{} ,GPU mem need: {}Mbyte", mem1, mem2, mem3, mem4, (mem1 + mem2 + mem3 + mem4) / (1024 * 1024));
        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let closures = program_closures!(
            |program, _arg| -> GPUResult<Vec<<G as PrimeCurveAffine>::Curve>> {
                let base_buffer = program.create_buffer_from_slice(bases)?;
                let exp_buffer = program.create_buffer_from_slice(exps)?;

                // It is safe as the GPU will initialize that buffer
                let bucket_buffer = unsafe {
                    program.create_buffer::<<G as PrimeCurveAffine>::Curve>(
                        2 * self.core_count * bucket_len,
                    )?
                };
                // It is safe as the GPU will initialize that buffer
                let result_buffer = unsafe {
                    program.create_buffer::<<G as PrimeCurveAffine>::Curve>(2 * self.core_count)?
                };

                // The global work size follows CUDA's definition and is the number of
                // `LOCAL_WORK_SIZE` sized thread groups.
                let global_work_size =
                    (num_windows * num_groups + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

                let kernel = program.create_kernel(
                    if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                        "G1_bellman_multiexp"
                    } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                        "G2_bellman_multiexp"
                    } else {
                        return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
                    },
                    global_work_size,
                    LOCAL_WORK_SIZE,
                )?;

                kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&exp_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                let mut results =
                    vec![<G as PrimeCurveAffine>::Curve::identity(); 2 * self.core_count];
                program.read_into_buffer(&result_buffer, &mut results)?;

                Ok(results)
            }
        );

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = acc.double();
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
        E: Engine + GpuEngine,
{
    kernels: Vec<SingleMultiexpKernel<E>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> MultiexpKernel<E>
    where
        E: Engine + GpuEngine,
{
    pub fn create(priority: bool) -> GPUResult<MultiexpKernel<E>> {
        let lock = locks::GPULock::lock();
        let lock_id = lock.id();

        let kernels: Vec<_> = Device::all()
            .iter()
            .filter_map(|device| {
                let kernel = SingleMultiexpKernel::<E>::create(device, priority);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                if device.unique_id() == lock_id {
                    kernel.ok()
                } else {
                    None
                }
            })
            .collect();

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        debug!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels.len(),
            get_cpu_utilization()
        );
        for (i, k) in kernels.iter().enumerate() {
            debug!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                i,
                k.program.device_name(),
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
        exps: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as PrimeCurveAffine>::Curve>
        where
            G: PrimeCurveAffine<Scalar=E::Fr>,
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
        //info!("cpu_utilization:{} , cpu_n={}, gpu_n={}", get_cpu_utilization(), cpu_n, n);

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;
        // println!("main MultiexpKernel.multiexp: exp_num:{} , num_devices:{} , chunk_size:{}",n,num_devices, chunk_size);

        let (tx_gpu, rx_gpu) = mpsc::channel();
        let (tx_cpu, rx_cpu) = mpsc::channel();
        let mut scoped_pool = Pool::new(2);
        scoped_pool.scoped(|scoped| {
            // GPU
            scoped.execute(move || {
                let start = Instant::now();
                debug!("scoped: gpu multiexp start...");
                let results = if n > 0 {
                    // println!("MultiexpKernel.multiexp: \n total bases.len():{},\n exps.len():{},\n chunk_size:{}",bases.len(),exps.len(),chunk_size);
                    bases
                        .par_chunks(chunk_size)
                        .zip(exps.par_chunks(chunk_size))
                        .zip(self.kernels.par_iter_mut())
                        .map(|((bases, exps), kern)| -> Result<<G as PrimeCurveAffine>::Curve, GPUError> {
                            let mut acc = <G as PrimeCurveAffine>::Curve::identity();
                            let single_chunk_size = (kern.n as f64 * (1 as f64 - get_cpu_utilization()) as f64).ceil() as usize;
                            let mut set_window_size = 11; //grouprate=>window_size : 2=>11,4=>11,8=>10,16=>9
                            let size_result = std::mem::size_of::<<G as PrimeCurveAffine>::Curve>();
                            if size_result > 144 {
                                set_window_size = 9; //grouprate=>window_size : 2=>8,4=>8,8=>8,16=>7
                            }
                            let mut times: u32 = 1;
                            for (bases, exps) in bases.chunks(single_chunk_size).zip(exps.chunks(single_chunk_size)) {
                                let now = Instant::now();
                                debug!("par[{}] Single multiexp start... ", times);
                                let result = kern.multiexp(bases, exps, bases.len(), set_window_size)?;
                                debug!("par[{}] Single multiexp end cost:{:?}", times, now.elapsed());
                                times += 1;
                                acc.add_assign(&result);
                            }
                            Ok(acc)
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                debug!("scoped: gpu multiexp end cost:{:?}", start.elapsed());
                tx_gpu.send(results).unwrap();
            });
            // CPU
            scoped.execute(move || {
                let start = Instant::now();
                debug!("scoped: cpu multiexp start...");
                let cpu_acc = cpu_multiexp::<_, _, _, E, _>(
                    &pool,
                    (Arc::new(cpu_bases.to_vec()), 0),
                    FullDensity,
                    Arc::new(cpu_exps.to_vec()),
                    &mut None,
                );
                debug!("scoped: cpu multiexp end cost:{:?}", start.elapsed());
                let cpu_r = cpu_acc.wait().unwrap();

                tx_cpu.send(cpu_r).unwrap();
            });
        });

        // waiting results...
        let results = rx_gpu.recv().unwrap();
        let cpu_r = rx_cpu.recv().unwrap();

        let mut acc = <G as PrimeCurveAffine>::Curve::identity();

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
