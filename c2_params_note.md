## SealCommitPhase1Output
```
SealCommitPhase1Output {
	registered_proof: StackedDrg32GiBV1_1,
	vanilla_proofs: StackedDrg32GiBV1([[..],[..],...,[..] 共10个 ]),
	comm_r: [int x32],
	comm_d: [int x32],
	replica_id: PoseidonDomain(0x HASHCODE),
	seed: [int x32],
	ticket: [int x32]
}
```

### vanilla_proofs

```
vanilla_proofs: StackedDrg32GiBV1(
	[
		[Proof {} x18] x10
	]
)
```
#### Proof

```
Proof {
	comm_d_proofs: MerkleProof {}
	comm_r_last_proof: MerkleProof {}
	replica_column_proofs: ReplicaColumnProof {}
	labeling_proofs: [LabelingProof {}x11
	]
	encoding_proof: EncodingProof {}
}
```

##### comm_d_proofs
```
comm_d_proofs {
	data: Single(
		SingleProof {
			root: Sha256Domain(HASHCODE)
			leaf: Sha256Domain(HASHCODE)
			path: InclusionPath {
				path: [
					PathElement {
						hashes: [Sha256Domain(HASHCODE)],
						index: 0,
						_arity: PhantomData
					} x30
				]
			}
		}
	)
}
```
##### comm_r_last_proof
```
comm_r_last_proof: MerkleProof {
	data: Sub(
		SubProof {
			root: PoseidonDomain(0x HASHCODE)
			leaf: PoseidonDomain(0x HASHCODE)
			base_proof: InclusionPath {
				path: [
					PathElement {
						hashes: [PoseidonDomain(0x HASHCODE) x7],
						index: 0,
						_arity: PhantomData

					} x7
				]
			},
			sub_proof: InclusionPath {
				path: [
					PathElement {
						hashes: [PoseidonDomain(0x HASHCODE) x7],
						index: 0,
						_arity: PhantomData
					}
				]
			},
		}
	)
}
```
##### replica_column_proofs
```
replica_column_proofs: ReplicaColumnProof {
	c_x: ColumnProof {
		column: Column {
			index: 284779691,
			rows: [PoseidonDomain(0x HASHCODE) x11],
			_h: PhantomData,
		}
		inclusion_proof: MerkleProof {} //同comm_r_last_proof:MerkleProof
	},
	drg_parents: [
		ColumnProof {}x6
	],
	exp_parents: [
		ColumnProof {}x8
	]
}
```
##### labeling_proofs

```
labeling_proofs: [
	LabelingProof {
		parents: [
			PoseidonDomain(0x HASHCODE) x37
		],
		layer_index: 1,
		node: 284779691,
		_h: PhantomData
	} x11
]
```

##### encoding_proof
```
encoding_proof: EncodingProof {
	parents: [
		PoseidonDomain(0x HASHCODE) x37
	],
	layer_index: 11,
	node: 284779691,
	_h: PhantomData,
}
```