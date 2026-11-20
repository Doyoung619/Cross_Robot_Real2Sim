# Descriptive evaluation analysis

These files analyze frozen final evaluations after adaptation. They are not inputs to training, retrieval, retargeting or checkpoint selection. The failure indicators overlap and are heuristic symptoms, not causal diagnoses: contact fraction <0.01; mean EEF tracking error >0.03 m; any logged joint-limit violation or arm/environment collision. Missing measurements are not classified as positive symptoms.

[Failure indicators](failure_indicators.csv) · [Stage ablations](stage_ablations.csv)

Ablation deltas compare the matched fixed-source control at the same allowed K and training seed; they are descriptive differences without a significance claim. The smoke horizon and two starts cannot establish adaptation quality. Check metrics.csv for actual counts, intervals and costs.

Cost bars show cumulative adaptation/training wall time and measured CUDA update timeline. The latter is not exclusive GPU-active time. Identical training cache hits cost zero actual new CUDA updates; equivalent uncached compute is separately retained in metrics.csv. Shared preparation cost remains in protocol.json.
