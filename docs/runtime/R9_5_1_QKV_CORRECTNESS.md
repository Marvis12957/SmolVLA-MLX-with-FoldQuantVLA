# R9.5.1 — QKV correctness hotfix investigation

## Decision

**NO FIX REQUIRED.** The suspected production bug is **NOT CONFIRMED**. No production runtime file was changed, because the locked artifact’s actual expert cross-attention path already supplies K and V with different cached inputs and independently applies each projection’s Fold transform. This report and its executable diagnostic are the persisted R9.5.1 deliverable.

Scope: `results/quantization/clean_track/artifacts/fq-pad768-language-downproj-w8/`, QKV sharing true, all compile flags false, ten Euler steps. The audit was performed on `1a3d02b7358fbdcdf10d9a4e667387a49941a632`.

## 1. Bug and exact execution trace

`_transformer` chooses self-attention on even expert layers and cross-attention on odd ones (`models/smolvla/src/smolvla_mlx/runtime.py:L728-L736`). On even expert layers, `_self_attention` normalizes the changing suffix hidden state and shares a prepared activation across Q/K/V only when all projections are `FoldLinear` (`L746-L812`). In the locked artifact, independent and shared Q/K/V outputs are exact for layers 0, 2, …, 14.

Odd expert layers do **not** enter the QKV-sharing branch. `_cross_attention` takes the independently cached language-prefix key and value tensors, flattens them separately, then executes:

```python
key = _heads(expert.k(flat_k), ...)
value = _heads(expert.v(flat_v), ...)
```

at `runtime.py:L830-L843`. `FoldLinear.__call__` applies that projection’s own padding, SmoothQuant scale, Hadamard transform, dynamic quantization/dequantization and dense inner projection (`runtime.py:L293-L325`). Therefore K receives `flat_k`; V receives `flat_v`. Neither cross projection calls `prepare_activation` or `forward_prepared`.

The alleged failure can be reproduced only by deliberately applying `expert.v.forward_prepared(expert.k.prepare_activation(flat_k))`. Across the eight odd layers this wrong control has V MAE 0.894–1.075 and max error 5.109–6.596. It is a valid illustration of why a future prefix-K/V optimization must use separate inputs/preparations, but is not the existing production dispatch.

## 2. Independent reference and parity

`models/smolvla/scripts/r9_5_1_qkv_correctness.py` loads fixed reference observations 0–2, explicit NumPy-generated `[1,50,32]` noise, and the locked artifact. It compares normal QKV sharing with the independent reference obtained by temporarily disabling `QKV_SHARING_ENABLED`; that makes every Q/K/V call use `FoldLinear.__call__` and consequently its correct own transform. It also directly compares even expert Q/K/V and SDPA outputs.

| Scope | Cosine | MAE | Max abs | Relative L2 |
|---|---:|---:|---:|---:|
| Full action, seeds 0, 1, 2 | 1.0 | 0.0 | 0.0 | 0.0 |
| Expert even Q/K/V + attention | 1.0 | 0.0 | 0.0 | 0.0 |
| Deliberately wrong odd V(K-input) control | -0.056 to 0.065 | 0.894–1.075 | 5.109–6.596 | 1.028–1.205 |

The raw per-layer results are `results/runtime/r9_5_1/pre_fix_parity.json`. “Pre-fix” means the audited, unmodified source; no fix was appropriate.

## 3. Regression checks

Executed:

- `uv run python models/smolvla/scripts/r9_5_1_qkv_correctness.py --benchmark --output results/runtime/r9_5_1/pre_fix_parity.json` — PASS. Covers all 16 expert layers, even Q/K/V sharing, odd K/V input distinction, 720→768 padded QKV, cache construction, three complete ten-step chunks, and 50-iteration timing.
- `uv run pytest models/smolvla/tests/test_prepare_activation_compile.py -q` — PASS, 8 tests.
- `uv run pytest models/smolvla/tests/test_prepare_activation_compile.py models/smolvla/tests/test_foldquant_int4.py -q` — 36 passed, 4 failed. The failures predate/noise outside R9.5.1: `QuantizedProjection` test arguments `activation_bits`/`input_pad` do not exist, a W8 kernel-name expectation predates `mlx_float`, and `FoldQuantInt4Linear.optimized` expectation is absent. No failure exercises or contradicts the cross-attention result.

## 4. KV-cache resolution

Instrumentation records 80 direct K and 80 direct V FoldLinear calls for odd expert cross-attention per policy inference: 8 odd layers × 10 Euler steps. Both have zero `prepare_activation` calls. Prefix cache contents remain invariant after crop, so precomputing projected K/V is mathematically plausible only if each projection uses its correct cached input and own transforms. R9.5.1 intentionally does not implement it.

## 5. Low-bit dispatch resolution

The loaded runtime has 224 `FoldLinear` projections and zero `nn.QuantizedLinear` projections. `_load_quantized` imports codes/scales (`runtime.py:L997-L1034`); `_projection` constructs `FoldLinear(..., _row_linear(...))` (`L883-L893`); `_row_linear` unpacks W4/constructs scaled float dense weights (`L336-L340`). `FoldLinear` dequantizes dynamic activation codes before calling its inner `nn.Linear` (`L187-L203`). Thus the locked artifact runs dense MLX floating arithmetic, not `FoldQuantInt4Linear` or `gemm_mlx_float32_w8`. The latter is available reference/experimental code in `runtime/mlx/metal/foldquant_linear.py:L132-L156`, but is not instantiated.

## 6. History and performance

R8.3/R8.4 identify `QKV_SHARING_ENABLED=true` and commit `1a3d02b`; their claimed path therefore includes this already-correct cross attention, not the hypothetical wrong V(K-input) path. Their published accuracy/latency require no correction from R9.5.1.

The matched diagnostic timing uses one fixed observation/noise, five warmups, explicit result synchronization and 50 measured complete chunks:

| Mode | p50 ms | p90 ms | mean ms |
|---|---:|---:|---:|
| Normal sharing | 194.93 | 195.66 | 195.40 |
| Independent reference (sharing disabled) | 198.62 | 199.09 | 198.41 |

This is a correctness comparison, not a production before/after improvement. Memory delta is not measured because no runtime implementation changed.

## 7. Git state

There is no runtime code diff: creating a production hotfix would falsely imply a repaired defect. The persisted diagnostic, JSON evidence and this report are committed under `R9.5.1 verify expert QKV sharing correctness`. Existing worktree modifications were preserved.
