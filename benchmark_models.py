import torch
import time
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Model configurations for reference
MODEL_CONFIGS = {
    "qwen3-8b": {
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "num_layers": 32,
        "description": "Qwen3-8B (GQA 32/8)",
    },
    "qwen3-30b": {
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "num_layers": 48,
        "description": "Qwen3-30B (GQA 40/8)",
    },
    "llama3-70b": {
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "num_layers": 80,
        "description": "Llama-3-70B (GQA 64/8)",
    },
    "qwen3-235b-moe": {
        "num_heads": 64,
        "num_kv_heads": 4,
        "head_dim": 128,
        "hidden_size": 8192,
        "num_layers": 94,
        "description": "Qwen3-235B-MoE (GQA 64/4, 128 experts)",
    },
}


def benchmark_prefill_at_scale(config_name: str, seq_lengths: list, batch_sizes: list):
    """Benchmark prefilling with flash attention at various scales."""
    from benchmark_prefilling import flash_attention, pytorch_standard_attention, setup_data

    config = MODEL_CONFIGS[config_name]
    print(f"\n{'='*70}")
    print(f"Prefilling Benchmark: {config['description']}")
    print(f"num_heads={config['num_heads']}, num_kv_heads={config['num_kv_heads']}, head_dim={config['head_dim']}")
    print(f"{'='*70}")

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            total_tokens = batch_size * seq_len

            # Skip configurations that would exceed memory
            estimated_memory_gb = (total_tokens * config['num_heads'] * config['head_dim'] * 2 * 3) / 1e9
            if estimated_memory_gb > 100:
                print(f"Skipping batch={batch_size}, seq={seq_len} (est. {estimated_memory_gb:.1f}GB)")
                continue

            print(f"\nbatch_size={batch_size}, seq_len={seq_len}, total_tokens={total_tokens}")

            try:
                q, k, v, cu_seqlens, scale = setup_data(
                    batch_size, seq_len,
                    config['num_heads'], config['num_kv_heads'], config['head_dim']
                )

                # Warmup
                for _ in range(3):
                    _ = flash_attention(q, k, v, cu_seqlens, scale,
                                       config['num_heads'], config['num_kv_heads'], config['head_dim'])

                torch.cuda.synchronize()

                # Benchmark
                num_iter = max(10, 100 // (seq_len // 256 + 1))
                start = time.perf_counter()
                for _ in range(num_iter):
                    _ = flash_attention(q, k, v, cu_seqlens, scale,
                                       config['num_heads'], config['num_kv_heads'], config['head_dim'])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / num_iter

                # Compute throughput
                tokens_per_sec = total_tokens / elapsed
                tflops = (2 * total_tokens * seq_len * config['num_heads'] * config['head_dim']) / elapsed / 1e12

                print(f"  Flash Attention: {elapsed*1000:.3f}ms | {tokens_per_sec/1e6:.2f}M tok/s | {tflops:.2f} TFLOPS")

                results.append({
                    'config': config_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'time_ms': elapsed * 1000,
                    'tokens_per_sec': tokens_per_sec,
                    'tflops': tflops,
                })

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM - skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")

    return results


def benchmark_decode_at_scale(config_name: str, context_lengths: list, batch_sizes: list):
    """Benchmark decoding with paged attention at various scales."""
    from benchmark_decoding import paged_attention_decode_triton, setup_test_data

    config = MODEL_CONFIGS[config_name]
    print(f"\n{'='*70}")
    print(f"Decoding Benchmark: {config['description']}")
    print(f"{'='*70}")

    block_size = 16
    results = []

    for batch_size in batch_sizes:
        for ctx_len in context_lengths:
            print(f"\nbatch_size={batch_size}, context_len={ctx_len}")

            try:
                q, k_cache, v_cache, block_tables, context_lens, scale = setup_test_data(
                    batch_size, ctx_len,
                    config['num_heads'], config['num_kv_heads'], config['head_dim'],
                    block_size
                )

                # Warmup
                for _ in range(5):
                    _ = paged_attention_decode_triton(
                        q, k_cache, v_cache, block_tables, context_lens,
                        scale, config['num_heads'], config['num_kv_heads'],
                        config['head_dim'], block_size
                    )

                torch.cuda.synchronize()

                # Benchmark
                num_iter = 100
                start = time.perf_counter()
                for _ in range(num_iter):
                    _ = paged_attention_decode_triton(
                        q, k_cache, v_cache, block_tables, context_lens,
                        scale, config['num_heads'], config['num_kv_heads'],
                        config['head_dim'], block_size
                    )
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / num_iter

                tokens_per_sec = batch_size / elapsed

                print(f"  Paged Attention: {elapsed*1000:.3f}ms | {tokens_per_sec:.0f} tokens/s")

                results.append({
                    'config': config_name,
                    'batch_size': batch_size,
                    'context_len': ctx_len,
                    'time_ms': elapsed * 1000,
                    'tokens_per_sec': tokens_per_sec,
                })

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM - skipping")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Large Model Attention Benchmark")
    parser.add_argument("--config", type=str, default="qwen3-8b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model configuration to benchmark")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["prefill", "decode", "all"],
                       help="Benchmark mode")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer configurations")
    args = parser.parse_args()

    print("="*70)
    print("Large Model Attention Benchmark")
    print("="*70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
    else:
        print("WARNING: CUDA not available, benchmarks will fail")
        return

    if args.quick:
        seq_lengths = [512, 2048]
        batch_sizes = [1, 4]
        context_lengths = [1024, 4096]
    else:
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        batch_sizes = [1, 2, 4, 8, 16]
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    if args.mode in ["prefill", "all"]:
        benchmark_prefill_at_scale(args.config, seq_lengths, batch_sizes)

    if args.mode in ["decode", "all"]:
        benchmark_decode_at_scale(args.config, context_lengths, batch_sizes)

    print("\n" + "="*70)
    print("Benchmark Complete")
    print("="*70)


if __name__ == "__main__":
    main()
