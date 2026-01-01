# MinivLLM fork

New Year 2026 LLM serving learning with a GB200 NVL72 cluster

A custom implementation of vLLM inference engine with attention mechanism benchmarks. Self-contained flash attention and paged attention implementations in Triton.

## GB200 Cluster Setup

This repository includes configurations for running on NVIDIA GB200 Grace Blackwell clusters:

- 4 nodes x 4 GPUs, NVL72
- 192GB HBM3e per GPU

## Quickstart (Local)

```bash
# Using conda
conda create -n minivllm python=3.11 -y
conda activate minivllm
pip install torch transformers xxhash triton

# Install package
pip install -e .

# Run benchmarks
python benchmark_prefilling.py
python benchmark_decoding.py
python main.py
```

Or with uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python main.py
```

## Benchmarks

### Prefilling (Flash Attention)

Compares attention implementations during prompt processing:

| Implementation | Memory | Notes |
|----------------|--------|-------|
| PyTorch Standard | O(N^2) | Materializes full attention matrix |
| Naive Triton | O(N^2) | Limited to 128 tokens by shared memory |
| Flash Attention | O(N) | Online softmax, block-wise processing |

```bash
python benchmark_prefilling.py
```

### Decoding (Paged Attention)

Compares implementations during token generation:

| Implementation | Description |
|----------------|-------------|
| Naive PyTorch | Loop-based with paged KV cache |
| Optimized PyTorch | Vectorized batch gathering |
| Triton Kernel | Custom paged attention kernel |

```bash
python benchmark_decoding.py
```

## Project Structure

```
MinivLLM/
├── src/myvllm/
│   ├── models/          # Model implementations (Qwen3)
│   ├── engine/          # LLM engine, scheduler, KV cache
│   ├── layers/          # Attention, MLP, embeddings
│   └── utils/           # Utilities
├── main.py              # Inference demo
├── benchmark_prefilling.py
├── benchmark_decoding.py
├── benchmark_large_model.py   # Large model configs
├── run_gb200_benchmark.sh     # Slurm job script
└── HowToApproachvLLM.md       # Implementation guide
```

## Requirements

- Python 3.11
- PyTorch with CUDA
- Triton
- transformers, xxhash

## References

- [vLLM](https://github.com/vllm-project/vllm)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Paged Attention](https://arxiv.org/abs/2309.06180)
