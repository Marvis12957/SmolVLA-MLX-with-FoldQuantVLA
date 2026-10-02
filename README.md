# SmolVLA MLX + FoldQuantVLA

## Pinned inputs

- **Checkpoint:** LeRobot pretrained `lerobot/smolvla_base`
  (`d9f33c94a60fb382c90dea2164c96845bd955e28`)
- **Local path:** `checkpoints/raw/smolvla/base`
- **Runtime target:** Apple M4 Max, 36 GB unified memory

No fine-tune. `smolvla-mlx fetch` downloads this checkpoint and inference uses it as published.

## Quick Start

### 1. Install dependencies
```bash
uv sync --all-extras
```

### 2. Fetch and convert
```bash
smolvla-mlx fetch
smolvla-mlx convert
```

### 3. Benchmark
```bash
smolvla-mlx verify --help
smolvla-mlx benchmark --help
```

## Workflow

```
1. Fetch `lerobot/smolvla_base`
2. Convert and validate the MLX artifact
3. Verify drift and benchmark latency, memory and model size
```

The present milestone is offline inference only. It does not fine-tune a
policy, control a physical robot, or claim that Metal uses an INT4 tensor-core
path.
