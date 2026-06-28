# Optimization Log

Running notes for performance work on the HIP runtime. Keep entries factual:
what changed, how it was measured, and what we learned.

## 2026-06-28

### Baseline Release RTF

Measured with `cargo run --release --bin hip-e2e-bench` using 39 generated frames,
3 measured iterations, and 1 warmup iteration.

| Model | Generation RTF | Decode RTF | E2E RTF |
| --- | ---: | ---: | ---: |
| 0.6B | `0.330165` | `0.102253` | `0.432418` |
| 1.7B | `0.414669` | `0.102953` | `0.517622` |

### CLI Timing Output

Added timing output to `hip-custom-voice-generate` so one-shot generation reports:

- model load time
- code generation time
- optional audio decode time
- optional WAV write time
- audio duration
- generation/decode/inference RTF when audio is decoded

This made quick command-line checks easier, but it is not a benchmark harness because
single runs include capture/warmup effects.

### HIP Graph Replay In CodePredictor

Tried graph replay in the production generation path. The first acoustic token stays
eager; the repeated remaining acoustic-group loop is captured once and replayed for
later frames.

Measured with `cargo run --release --bin hip-e2e-bench`.

| Model | Before E2E RTF | After E2E RTF | Change |
| --- | ---: | ---: | ---: |
| 0.6B | `0.432418` | `0.421308` | ~2.6% faster |
| 1.7B | `0.517622` | `0.505415` | ~2.4% faster |

0.6B after graph replay:

```text
generation_rtf=0.319113, decode_rtf=0.102195, e2e_rtf=0.421308
```

Discovery: HIP Graph replay helps, but only modestly. Launch overhead is not the main
bottleneck in the current generation path.

### ROCm Profiler Availability

Tried to use `rocprofv3`, but the installed ROCm image does not include the profiler
CLI in `/opt/rocm/bin` or `/opt/rocm-7.2.4/bin`. The install currently has profiler
registration libraries only.

Because of that, added a synchronized stage profile to `hip-rollout-bench` for coarse
breakdown without external profiler tooling.

### Rollout Stage Profile

Measured 0.6B, 39 frames, release build, fixture-backed rollout.

```text
prefill_seconds=0.009874
prepare_prefix_seconds=0.001644
code_predictor_seconds=0.712256
build_step_input_seconds=0.001654
talker_decode_seconds=0.269924
total_seconds=0.995351
```

| Stage | Share |
| --- | ---: |
| CodePredictor | `71.56%` |
| Talker decode | `27.12%` |
| Prefill | `0.99%` |
| Prepare prefix | `0.17%` |
| Build step input | `0.17%` |

Discovery: generation is dominated by `HipCodePredictor`, not talker decode and not
prefix/step-input glue.

### Low-Level Decode-Step Graph Benchmarks

Measured graph replay on the lower-level `DecodeStepStack` path.

```text
5-layer CodePredictor-sized stack: graph_speedup=1.067
28-layer talker-sized stack:       graph_speedup=1.056
```

Discovery: even at the transformer-stack level, graph replay only gives about 5-7%.
This matches the small end-to-end gain and points away from launch overhead as the
main bottleneck.

### Current Conclusion

The next serious optimization target should be CodePredictor math efficiency,
especially the many single-token/small-matrix operations used while predicting the
15 acoustic groups per frame.

Likely next experiments:

- replace small `m=1` rocBLAS SGEMMs with custom GEMV kernels
- fuse GEMV-adjacent operations where practical, such as bias/norm/argmax pieces
- profile CodePredictor sub-stages more deeply if ROCm profiler tooling becomes available
- evaluate fp16/bf16 paths later, after exact f32 path remains stable

### CodePredictor Sub-Stage Profile

Added a diagnostic profile path for `HipCodePredictor` and called it from
`hip-rollout-bench`. Measured 0.6B, 39 frames, release build.

```text
code_predictor_seconds=0.813764
prefix_projection_seconds=0.000007
stack_prefill_seconds=0.057901
first_logits_seconds=0.003347
first_token_seconds=0.001633
remaining_projection_seconds=0.000094
remaining_stack_seconds=0.679855
remaining_logits_seconds=0.046197
remaining_token_seconds=0.022536
output_copy_seconds=0.001217
```

Discovery: CodePredictor time is dominated by the transformer stack, especially the
remaining acoustic-group decode steps. LM-head logits and token/embedding glue are
secondary.

### Naive Custom GEMV Experiment

Added `gemv-bench` to compare rocBLAS `m=1` SGEMM against a simple custom one-block
per-output-column GEMV kernel.

| Shape `(n, k)` | rocBLAS Mean | Custom GEMV Mean | Result |
| --- | ---: | ---: | ---: |
| `(1024, 1024)` | `27.833 us` | `94.379 us` | custom slower |
| `(2048, 1024)` | `32.386 us` | `158.965 us` | custom slower |
| `(6144, 1024)` | `164.140 us` | `323.879 us` | custom slower |

Discovery: a naive f32 GEMV replacement is not competitive with rocBLAS. If we replace
rocBLAS, it needs a more specialized/tiled kernel, not the simple reduction kernel.

### Decode-Step Sub-Stage Profile

Added a diagnostic `DecodeStepStack::decode_step_profiled` path and printed it from
`decode-step-graph-bench`.

Before removing decode identity permutes, CodePredictor-sized 5-layer stack:

```text
total_seconds=0.002605
input_norm=0.000147
qkv_gemm=0.000350
qk_layout_cache=0.000268
attention=0.000205
output_gemm_residual=0.000399
post_norm=0.000139
gate_up_gemm=0.000392
swiglu=0.000128
down_gemm_residual=0.000548
final_copy=0.000029
```

Discovery: decode-step cost is spread across GEMMs, with the MLP down projection and
attention output projection among the larger buckets. The non-GEMM q/k layout/cache
bucket was also worth checking.

### Remove Single-Token Decode Permutes

In decode-step paths, `permute_bshd_to_bhsd` is an identity when `steps=1`. Removed
the q/k/v permute launches from eager, stream/graph, and profiled decode-step paths;
prefill paths still keep the real layout transform.

Measured with `cargo run --release --bin hip-e2e-bench`.

| Model | Before E2E RTF | After E2E RTF | Generation RTF After |
| --- | ---: | ---: | ---: |
| 0.6B | `0.421308` | `0.412695` | `0.309310` |
| 1.7B | `0.505415` | `0.497384` | `0.393780` |

Discovery: removing the no-op permutes produced another small but real win, roughly
2% e2e. Combined with CodePredictor graph replay, 0.6B moved from about `0.432` to
about `0.413` e2e RTF.

### CodePredictor Stream Sync Copy

Tested removing the conservative one-token D2H copy before CodePredictor graph replay.
Quick parity passed, but 0.6B e2e did not improve (`0.413102` RTF in the test run), so
the explicit sync copy was kept for safer stream ordering.

### Python/HIP WAV Comparison And Repetition Penalty

Compared Python WAV output against HIP output for:

```text
She said she would be here by noon.
speaker=Ryan
language=English
```

The HIP path is not sample-rate slow: both Python and HIP write `24000 Hz` WAVs.
The apparent differences came from generation settings, not WAV sample rate.

| Output | Frames | Duration | Notes |
| --- | ---: | ---: | --- |
| Python default high-level | `48` | `3.84s` | non-streaming, sampled subtalker, repetition penalty |
| Python non-streaming argmax | `45` | `3.60s` | non-streaming, deterministic subtalker |
| Python streaming argmax | `39` | `3.12s` | streaming mode, repetition penalty `1.05` |
| Python streaming argmax, repetition penalty `1.0` | `39` | `3.12s` | matches HIP default codes |
| HIP default | `39` | `3.12s` | streaming-style, argmax, repetition penalty `1.0` |

Discovery: the project can keep streaming mode as the primary behavior. Python parity
for this mode requires making semantic repetition penalty configurable. Added
`GenerateOptions::repetition_penalty`, initially defaulting to `1.0` so existing
streaming argmax behavior and parity stayed unchanged. The public default was later
changed to `1.05` after matching Python streaming argmax with repetition penalty;
parity scripts explicitly pass `1.0` for the original no-penalty fixtures.

Validation:

```text
HIP repetition_penalty=1.05 matched Python streaming argmax codes exactly.
python_stream_1p05 duration=3.12s rms=0.050849 peak=0.503632
hip_stream_1p05    duration=3.12s rms=0.050840 peak=0.503632
```

### Qwen Generation Defaults

`GenerateOptions::default()` now follows the Qwen TTS wrapper generation defaults while
keeping this project's streaming path:

```text
do_sample=true
top_k=50
top_p=1.0
temperature=0.9
repetition_penalty=1.05
subtalker_dosample=true
subtalker_top_k=50
subtalker_top_p=1.0
subtalker_temperature=0.9
seed=0
```

The parity script explicitly passes `do_sample=false`, `subtalker_dosample=false`, and
`repetition_penalty=1.0` because the stored fixtures test the deterministic greedy path.

Measured hot e2e RTF for the streaming/Qwen-default path with 3 measured iterations
and 1 warmup iteration, excluding model load:

| Runtime | Frames | Audio Seconds | Generation RTF | Decode RTF | E2E RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| HIP `HipTtsEngine` | `34` | `2.72` | `0.440904` | `0.097584` | `0.538488` |
| Python reference | `46` | `3.68` | `1.755115` | `0.150108` | `1.905224` |

These use the same streaming/Qwen-default generation settings, but sampling means the
exact output lengths can differ unless the sampling RNGs are matched exactly.
