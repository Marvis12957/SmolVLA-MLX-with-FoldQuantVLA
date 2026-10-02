# Hardware-Aware FoldQuant for SmolVLA on Apple Silicon / MLX

## 0. Mission

Build an Apple-Silicon-native quantization and inference pipeline for SmolVLA whose primary objective is:

> Reduce model size and unified-memory requirements as far as possible while preserving useful action/task accuracy, without introducing transformations that negate the performance advantages of MLX/Metal.

This project is **not** a direct port of FoldQuantVLA.

The research target is:

**Hardware-Aware FoldQuant for Apple Silicon / MLX**

The core hypothesis is:

> Fold/rotation/scaling can reshape SmolVLA into a representation that is more tolerant of aggressive MLX-native low-bit quantization. This may allow a lower effective bit-width than plain MLX quantization at the same accuracy, or higher accuracy at the same memory budget.

The final comparison must therefore be:

```text
BF16/FP16 SmolVLA
        ↓
Plain MLX quantization
        vs.
Fold-aware MLX quantization
        ↓
accuracy / model size / unified memory / latency
```

The project succeeds only if Fold-aware quantization improves the Pareto frontier.

---

# 1. Research questions

The implementation must answer these questions experimentally.

### RQ1 — How far can SmolVLA be compressed on Apple Silicon?

Test at minimum:

- BF16/FP16
- MLX W8
- MLX W6
- MLX W4
- MLX W3
- MLX W2 where supported
- mixed W2/W3/W4/W8/FP16
- NVFP4
- optional W4A4/NVFP4 through MLX QQLinear

Do not assume W4 is the final target.

The target is the **lowest effective bits per parameter that remains useful**.

---

### RQ2 — Does FoldQuant-style transformation actually help MLX quantization?

Compare at identical quantization settings:

```text
MLX W4
vs
Fold + MLX W4

MLX W3
vs
Fold + MLX W3

MLX W2
vs
Fold + MLX W2
```

A valid contribution exists if Fold allows either:

```text
same size  -> better accuracy
```

or:

```text
same accuracy -> lower bit width / memory
```

or ideally both.

---

### RQ3 — Which SmolVLA layers are sensitive to aggressive quantization?

Do not blindly copy the FoldQuantVLA selective-INT8 recipe.

Use `o_proj` and MLP output/down projections as **priors only**.

Measure SmolVLA itself.

Determine sensitivity separately for:

- vision encoder
- multimodal connector
- VLM attention
- VLM MLP
- action expert attention
- action expert MLP
- state projection
- action input projection
- action output projection
- timestep/action MLPs
- embeddings
- normalization layers

---

### RQ4 — Is runtime Fold transformation worth its cost on Apple Silicon?

A transformation that improves quantization error but adds excessive Metal latency is not acceptable.

For every transform measure:

```text
accuracy gain
memory gain
transform latency
total end-to-end latency
```

The transform must be treated as part of inference cost.

---

# 2. Hard constraints

## Apple Silicon first

Primary runtime:

```text
macOS
Apple Silicon
MLX
Metal
Unified Memory
batch size = 1
```

Do NOT design around:

- CUDA
- TensorRT
- CUTLASS
- NVIDIA INT4 tensor cores
- CUDA-only GPTQ kernels

Offline calibration may use PyTorch if necessary, but the deployed inference path must be MLX-native.

---

## Prefer existing MLX kernels

Priority order:

```text
1. MLX native operation
2. MLX compiled graph
3. existing optimized Metal operation
4. custom Metal kernel only if absolutely necessary
```

Do not implement custom Metal kernels during the first research phase.

In particular, investigate and benchmark MLX-native:

```python
mlx.nn.QuantizedLinear
mlx.core.quantized_matmul
mlx.core.quantize
mlx.core.hadamard_transform
mlx.nn.QQLinear
mlx.core.qqmm
```

before creating custom kernels.

---

# 3. Important scientific constraint

The original FoldQuantVLA recipe must NOT be assumed to work on SmolVLA.

The paper itself reports poor exploratory SmolVLA W4A4 results.

Therefore treat the project as:

```text
FoldQuant principles
        +
SmolVLA-specific sensitivity analysis
        +
MLX-native quantization
        +
Apple-Silicon hardware constraints
```

not:

```text
copy FoldQuantVLA implementation
        +
replace TensorRT with MLX
```

---

# 4. Repository structure

Create a clean research implementation.

```text
project/
├── README.md
├── PLAN.md
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── w4.yaml
│   ├── w3.yaml
│   ├── mixed.yaml
│   └── fold_w4.yaml
│
├── src/
│   └── foldmlx/
│       ├── model/
│       │   ├── smolvla_mlx.py
│       │   ├── vlm.py
│       │   ├── expert.py
│       │   └── loader.py
│       │
│       ├── quant/
│       │   ├── mlx_quant.py
│       │   ├── mixed_precision.py
│       │   ├── sensitivity.py
│       │   └── recipes.py
│       │
│       ├── fold/
│       │   ├── graph.py
│       │   ├── scaling.py
│       │   ├── hadamard.py
│       │   ├── folding.py
│       │   └── calibration.py
│       │
│       ├── eval/
│       │   ├── fidelity.py
│       │   ├── latency.py
│       │   ├── memory.py
│       │   └── benchmark.py
│       │
│       └── utils/
│
├── scripts/
│   ├── convert_smolvla_to_mlx.py
│   ├── collect_calibration.py
│   ├── analyze_activations.py
│   ├── quantize.py
│   ├── benchmark.py
│   ├── evaluate_fidelity.py
│   └── search_mixed_precision.py
│
├── tests/
│   ├── test_conversion.py
│   ├── test_fold_equivalence.py
│   ├── test_quantization.py
│   └── test_inference.py
│
└── results/
    ├── baseline/
    ├── quantization/
    ├── ablations/
    └── pareto/
```

Do not tightly couple experiment code to one checkpoint.

---

# 5. Phase 0 — Environment and hardware audit

Before implementing quantization, record the target hardware.

Output:

```text
results/system.json
```

Record at minimum:

```json
{
  "chip": "...",
  "memory_gb": 0,
  "macos": "...",
  "mlx_version": "...",
  "python": "...",
  "metal_available": true
}
```

Also programmatically check which MLX quantization modes are actually available in the installed version.

Do not assume API availability based only on documentation.

---

# 6. Phase 1 — Establish PyTorch reference

Use the official LeRobot SmolVLA implementation as the reference model.

Default checkpoint:

```text
lerobot/smolvla_base
```

Create deterministic reference observations.

For each observation save:

```text
images
language tokens
language mask
state
noise
flow-matching timestep/noise schedule
final action chunk
```

Use fixed random seeds.

The exact same observation and noise must be used when comparing all models.

Create:

```text
reference_inputs/
reference_outputs/
```

The PyTorch reference must never be silently changed during later experiments.

---

# 7. Phase 2 — Implement FP16/BF16 SmolVLA in MLX

Before quantization, obtain an MLX model that reproduces the original policy closely.

Port/reuse:

```text
vision encoder
multimodal connector
language/VLM blocks
action expert
KV cache
RoPE
RMSNorm
attention
MLP
state projection
action projections
flow-matching inference loop
```

Reuse existing MLX components wherever compatible.

Do not optimize aggressively yet.

First objective:

```text
PyTorch BF16
vs
MLX BF16/FP16
```

Compare final decoded action chunks.

Record:

```text
cosine similarity
RMSE
MAE
max absolute error
per-coordinate error
```

Target:

```text
median cosine >= 0.999
```

Treat this as a target rather than hiding numerical differences if MLX and PyTorch kernels produce slightly different results.

Do not proceed to quantization until the baseline discrepancy is understood.

---

# 8. Phase 3 — Establish plain MLX quantization baselines

This phase is critical.

No Fold transforms are allowed yet.

Quantize eligible Linear layers using native MLX quantization.

Test:

```text
W8
W6
W5
W4
W3
W2
NVFP4
MXFP4 where appropriate
```

At minimum explore affine group sizes:

```text
32
64
128
```

when supported.

For every configuration record:

```text
checkpoint size
quantized parameter bytes
effective bits/parameter
peak unified memory
active MLX memory
cold-start latency
warm latency
action-chunk latency
action fidelity
task success
```

Generate:

```text
results/quantization/plain_mlx.csv
```

This table is the baseline the proposed method must beat.

---

# 9. Do not confuse W4 with W4A4

Use precise naming.

If using:

```python
mlx.nn.QuantizedLinear
```

with FP16/BF16 activations, call it:

```text
W4A16
W3A16
W2A16
```

as appropriate.

Do NOT call it W4A4.

Only call a configuration W4A4 when activation quantization is actually performed.

---

# 10. Phase 4 — Activation and layer sensitivity analysis

Instrument the FP model.

Collect activation statistics from calibration observations.

For every candidate projection record:

```text
max(abs(x))
mean(abs(x))
median(abs(x))
percentiles
channel-wise maxima
outlier ratio
activation norm
weight norm
quantization reconstruction error
```

Compute an outlier score such as:

```text
max_channel_amplitude / median_channel_amplitude
```

Then perform **one-layer-at-a-time quantization**.

Example:

```text
all FP16
↓
quantize only layer i
↓
measure action degradation
```

Produce:

```text
results/sensitivity/layers.csv
```

Columns:

```text
layer
module_type
parameter_count
activation_outlier_score
W4_error
W3_error
W2_error
action_cosine_drop
action_rmse_delta
latency_contribution
```

Use this data to identify genuinely sensitive SmolVLA layers.

---

# 11. Phase 5 — Implement Fold transformation

Implement FoldQuant-style transformations independently of quantization.

For a projection:

```text
y = W x + b
```

use transformed activation:

```text
z = T x
```

and folded weight:

```text
W_f = W T^-1
```

such that before quantization:

```text
W_f z + b ≈ W x + b
```

and mathematically should be equal up to numerical precision.

Start with:

```text
T = R S^-1
```

where:

```text
S = diagonal channel scaling
R = orthogonal Hadamard transform
```

Corresponding fold-before weight:

```text
W_f = W S R^T
```

---

# 12. Consistent folding is mandatory

Create an activation-site graph.

If activation `x` feeds multiple consumers:

```text
x
├── Wq
├── Wk
└── Wv
```

all consumers must share the same transformation `T`.

Do NOT independently calibrate three incompatible transforms.

Runtime should conceptually be:

```text
x
↓
T(x)
↓
shared transformed activation
├── quantized Wq_f
├── quantized Wk_f
└── quantized Wv_f
```

Add a unit test proving that consistent folding reproduces the FP projection before quantization.

---

# 13. Runtime transform strategy for MLX

The major difference from the original FoldQuantVLA implementation is that runtime transformation must be justified on Apple Silicon.

Implement transformations in increasing order of cost.

### Level A — Offline/absorbed scaling

Prefer transforms whose cost can be absorbed into:

```text
RMSNorm gain
producer weights
adjacent compatible linear layers
```

Goal:

```text
zero additional runtime kernel
```

Use this whenever mathematically valid.

---

### Level B — Diagonal runtime scaling

If scaling cannot be absorbed:

```python
z = x * scale
```

Benchmark its actual cost.

---

### Level C — Hadamard rotation

Use MLX's native Hadamard operation.

For FoldQuant-style block Hadamard:

```text
reshape (..., hidden_dim)
→ reshape (..., num_blocks, 64)
→ hadamard_transform(last_axis)
→ reshape back
```

Do not construct a dense Hadamard matrix.

Never implement:

```python
x @ H
```

with a dense matrix.

Benchmark block sizes:

```text
32
64
128
```

where shapes allow.

Start with 64 because it matches the FoldQuantVLA design, but do not assume it is optimal on Apple hardware.

---

# 14. Runtime-transform acceptance rule

A Fold transform is not automatically useful because it improves cosine similarity.

Calculate:

```text
total_latency =
    transform_latency
    + quantized_projection_latency
    + remaining_model_latency
```

Reject transformations that improve fidelity but make the complete policy materially slower unless they enable a lower bit width with meaningful memory reduction.

Target:

```text
Fold runtime overhead < 5% end-to-end
```

Prefer <2%.

---

# 15. Phase 6 — Fold + MLX quantization

After the FP equivalence test passes:

```text
original W
↓
fold transform
↓
W_f
↓
MLX quantization
```

Test:

```text
Fold-W4A16
Fold-W3A16
Fold-W2A16
Fold-NVFP4
```

Compare directly against:

```text
MLX-W4A16
MLX-W3A16
MLX-W2A16
MLX-NVFP4
```

Same:

```text
checkpoint
calibration observations
evaluation observations
noise
task episodes
group size
bit width
hardware
```

must be used.

---

# 16. Phase 7 — MLX-native activation quantization

Do this only after the weight-only path is stable.

Investigate MLX QQLinear / QQMM.

Primary experiment:

```text
NVFP4 weight
+
NVFP4 activation
```

Do not reproduce the TensorRT/CUTLASS INT4 path from FoldQuantVLA.

Use the MLX-native path.

Because QQLinear may have API limitations such as bias handling, implement a wrapper only if needed:

```text
y = QQLinear(x)
y = y + bias
```

but benchmark the extra operation.

Compare:

```text
W4A16
vs
W4A4
```

for:

```text
plain MLX
Fold + MLX
```

The key question is whether Fold reduces activation-quantization damage enough to justify W4A4 on SmolVLA.

---

# 17. Phase 8 — Mixed precision search

The final model does NOT need every layer to use the same precision.

Search for:

```text
W2
W3
W4
W6
W8
FP16
```

per layer/module group.

Objective:

```text
minimize total model bytes
```

subject to:

```text
action fidelity >= threshold
task success >= threshold
latency <= threshold
```

Start from the lowest precision and selectively rescue sensitive layers.

Example:

```text
most linears              W3
sensitive o_proj          W4/W8
sensitive MLP output      W4/W8
action_out_proj           W8/FP16
normalization             FP16
small layers              FP16 if quantization saves negligible memory
```

Do not copy this exact recipe blindly.

The sensitivity profiler must determine the final recipe.

---

# 18. Greedy mixed-precision algorithm

Implement a simple reproducible search first.

Pseudo algorithm:

```python
recipe = quantize_everything(lowest_bits)

while accuracy(recipe) < target:
    for candidate_layer in remaining_layers:
        test upgrade candidate:
            W2 -> W3
            W3 -> W4
            W4 -> W8
            W8 -> FP16

        score =
            accuracy_recovery /
            added_model_bytes

    apply best upgrade
```

Also include latency in a second scoring mode:

```text
score =
accuracy_recovery /
(alpha * added_bytes + beta * added_latency)
```

Save every search step.

---

# 19. Accuracy metrics

Do not rely on cosine similarity alone.

FoldQuantVLA itself demonstrates that high action cosine does not guarantee task success.

Use two levels.

## Level 1 — Offline action fidelity

Use held-out observations.

Measure:

```text
cosine similarity
minimum cosine
median cosine
RMSE
MAE
max absolute coordinate error
per-action-dimension error
```

Use identical flow-matching noise.

---

## Level 2 — Closed-loop task performance

Run actual simulation/robot benchmark when available.

Primary metric:

```text
task success rate
```

Report:

```text
success / total episodes
```

and confidence intervals where practical.

Offline fidelity is a screening metric.

Task success is the final accuracy metric.

---

# 20. Hardware metrics

The project is hardware-aware, so accuracy alone is insufficient.

Measure all of:

```text
model file size
quantized weight bytes
metadata/scales bytes
effective bits per parameter
peak MLX memory
active MLX memory
process resident memory where available
load time
prefill latency
denoising/action generation latency
full action-chunk latency
```

Use MLX memory APIs where possible.

Always warm up before latency measurement.

Example protocol:

```text
10 warmups
100 measured iterations
batch = 1
```

Report:

```text
median
p50
p90
p95
std
```

Do not report only one lucky timing run.

---

# 21. Effective bits per parameter

Implement:

```text
effective_bpw =
8 * total_quantized_model_bytes /
number_of_original_parameters
```

Also report separately:

```text
weight payload
scales
zero-points/bias metadata
remaining FP parameters
```

This prevents misleading claims such as:

```text
"4-bit model = exactly 4 bits/parameter"
```

when metadata and FP layers are present.

---

# 22. Primary experiment table

The final paper should contain a table approximately like:

```text
Method              Size   BPW   Peak Mem   Latency   Fidelity   Success
---------------------------------------------------------------------------
PyTorch BF16
MLX FP16
MLX W8
MLX W4
MLX W3
MLX W2
Fold-MLX W4
Fold-MLX W3
Fold-MLX W2
Mixed MLX
Fold-Mixed MLX
Fold-MLX W4A4
```

The key rows are:

```text
MLX W3      vs Fold-MLX W3
MLX W2      vs Fold-MLX W2
Mixed MLX   vs Fold-Mixed MLX
```

because these demonstrate whether Fold contributes anything beyond MLX's existing quantizer.

---

# 23. Required ablations

Run at minimum:

```text
plain MLX
+ scaling only
+ Hadamard only
+ scaling + Hadamard
+ consistent shared transform
+ intentionally independent/mismatched transform for diagnostic only
```

Then:

```text
W4
W3
W2
```

and:

```text
uniform precision
vs
mixed precision
```

Also compare:

```text
no runtime rotation
vs
runtime Hadamard
```

to establish whether rotation is worth its hardware cost.

---

# 24. Quantization modes to prioritize

Recommended experiment order:

```text
1. affine W4 group=64
2. affine W3 group=64
3. affine W2 group=64
4. affine W4 group=32
5. NVFP4
6. Fold + affine W4
7. Fold + affine W3
8. Fold + affine W2
9. Fold + NVFP4
10. mixed precision
11. QQLinear W4A4/NVFP4
```

Do not start with W4A4.

First establish whether Fold improves weight quantization.

---

# 25. Compression objective

Do not optimize only for checkpoint size.

Define the main objective as a Pareto frontier:

```text
maximize:
    task accuracy

minimize:
    effective bits/parameter
    checkpoint size
    peak unified memory
    action latency
```

A configuration is interesting when no other configuration is simultaneously:

```text
smaller
faster
and
more accurate
```

Generate:

```text
results/pareto/accuracy_vs_size.png
results/pareto/accuracy_vs_memory.png
results/pareto/accuracy_vs_latency.png
```

---

# 26. Definition of the research contribution

The desired contribution is NOT:

> We ran SmolVLA using MLX.

It is NOT:

> We quantized SmolVLA to 4-bit.

It should eventually be supportable as:

> We introduce a hardware-aware folding and mixed-precision quantization pipeline for deploying VLA models on Apple Silicon. By aligning activation representations with MLX-native low-bit kernels and selectively retaining precision only in sensitive projections, the method reduces model and unified-memory requirements while preserving useful policy fidelity.

Do not write this as a confirmed result until experiments support it.

The strongest possible result would be:

```text
Plain MLX W4
    accuracy X
    size Y

Fold-aware MLX W3
    accuracy ≈ X
    size < Y
```

or:

```text
Plain MLX W3
    unacceptable accuracy

Fold-aware MLX W3
    acceptable accuracy
```

This would directly demonstrate that Fold makes the model more quantization-friendly on the target hardware.

---

# 27. Acceptance gates

## Gate A — MLX parity

Do not quantize until:

```text
MLX FP model ≈ PyTorch reference
```

---

## Gate B — Plain quantization baseline

Do not claim Fold improvement until plain MLX W8/W4/W3/W2 baselines exist.

---

## Gate C — Fold mathematical correctness

Before quantization:

```text
Folded FP output ≈ original FP output
```

Unit tests must verify this.

---

## Gate D — Hardware benefit

Do not retain a Fold transform merely because reconstruction error improves.

It must improve the accuracy-memory-latency tradeoff.

---

## Gate E — Research contribution

Before claiming success, demonstrate at least one of:

```text
Fold same bits > plain MLX accuracy

Fold lower bits ≈ plain MLX higher-bit accuracy

Fold mixed precision < plain MLX memory at similar accuracy
```

---

# 28. Coding rules for the agent

- Keep PyTorch reference code separate from MLX deployment code.
- Never silently fall back from Metal to CPU.
- Assert Metal availability in benchmark scripts.
- Keep all experiments deterministic where possible.
- Save every config with every result.
- Save git commit hash with experiment output.
- Never overwrite previous experiment results.
- Do not hardcode layer names until model inspection confirms them.
- Do not quantize normalization layers by default.
- Do not quantize tiny layers when metadata overhead exceeds meaningful savings.
- Do not introduce custom Metal kernels until native MLX paths have been benchmarked.
- Do not claim W4A4 unless activations are actually 4-bit.
- Do not use action cosine as the sole quality metric.
- Do not modify the original model architecture merely to make quantization look better.
- Any architecture change must be a separate ablation.

---

# 29. First implementation milestone

The first PR should contain ONLY:

```text
1. environment audit
2. PyTorch reference runner
3. MLX FP16/BF16 SmolVLA
4. deterministic action comparison
5. latency benchmark
6. memory benchmark
```

No FoldQuant yet.

Expected command:

```bash
python scripts/benchmark.py \
    --model smolvla \
    --backend mlx \
    --precision fp16
```

and:

```bash
python scripts/evaluate_fidelity.py \
    --reference pytorch \
    --candidate mlx-fp16
```

---

# 30. Second milestone

Implement native MLX quantization only.

Expected CLI:

```bash
python scripts/quantize.py \
    --bits 4 \
    --group-size 64 \
    --mode affine \
    --output checkpoints/mlx-w4
```

Then support:

```bash
--bits 3
--bits 2
--mode nvfp4
```

Produce the complete plain-MLX baseline table.

---

# 31. Third milestone

Implement calibration + Fold.

Expected pipeline:

```bash
python scripts/collect_calibration.py

python scripts/analyze_activations.py

python scripts/quantize.py \
    --fold \
    --rotation hadamard \
    --block-size 64 \
    --bits 4 \
    --group-size 64
```

Then repeat for W3 and W2.

---

# 32. Fourth milestone

Implement automatic mixed precision.

Example:

```bash
python scripts/search_mixed_precision.py \
    --base-bits 2 \
    --target-fidelity 0.99 \
    --objective memory
```

The script must produce a machine-readable recipe:

```json
{
  "default": {
    "bits": 3,
    "group_size": 64
  },
  "layers": {
    "...o_proj": {
      "bits": 4
    },
    "...action_out_proj": {
      "bits": 8
    }
  }
}
```

The exact layer choices must come from experiments.

---

# 33. Fifth milestone — W4A4 experiment

Only after all previous milestones pass:

```text
Fold
+
MLX QQLinear / QQMM
+
NVFP4 activation
+
NVFP4 weight
```

Benchmark against W4A16.

If W4A4:

```text
is slower
and
does not materially reduce memory
and/or
damages task success
```

do not use it in the final deployment recipe.

Hardware-aware optimization means choosing what actually works on Apple Silicon, not forcing W4A4 because the original FoldQuant paper used it on NVIDIA hardware.

---

# 34. Final deliverables

The repository must eventually produce:

```text
SmolVLA MLX FP baseline
plain MLX low-bit models
Fold-aware MLX models
mixed-precision recipe
calibration tool
sensitivity profiler
memory benchmark
latency benchmark
action-fidelity benchmark
task evaluation
Pareto plots
reproducible experiment configs
```

The final artifact should allow:

```bash
python scripts/quantize.py \
    --model lerobot/smolvla_base \
    --recipe configs/fold_mixed.yaml \
    --output smolvla-foldmlx
```

followed by:

```bash
python scripts/benchmark.py \
    --model smolvla-foldmlx
```

---

# 35. Immediate task for the coding agent

Start with Phase 0–2 only.

Before writing significant code:

1. Inspect the current SmolVLA/LeRobot implementation.
2. Print the full module tree and parameter shapes.
3. Calculate parameter counts by subsystem.
4. Identify VLM, vision, expert, and action-projection boundaries.
5. Inspect the installed MLX quantization API.
6. Build the deterministic PyTorch reference.
7. Port or adapt the minimum MLX inference implementation.
8. Verify FP parity.
9. Measure baseline model size, unified-memory usage, and latency.
10. Commit these results before implementing quantization.

Do **not** start by implementing FoldQuant.

We need a trustworthy baseline first.

---

# 36. Core principle

Every design decision must answer:

> Does this make SmolVLA smaller or cheaper to run on a Mac while preserving enough policy quality?

If not, remove it.

The desired path is:

```text
SmolVLA
   ↓
analyze hardware + activation sensitivity
   ↓
Fold only where useful
   ↓
MLX-native aggressive quantization
   ↓
selectively restore precision
   ↓
smallest acceptable model
   ↓
Apple-Silicon-native deployment
```

The target is not "maximum quantization".

The target is:

> **minimum hardware cost at acceptable policy quality.**