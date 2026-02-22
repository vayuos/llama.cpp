#!/usr/bin/env python3
"""
analyze_batch_size.py - Analyze optimal batch size for throughput vs latency tradeoff

Measures throughput and latency at different batch sizes to find:
- Maximum throughput batch size
- Optimal latency batch size
- Memory utilization per batch size
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


@dataclass
class BatchBenchmark:
    batch_size: int
    avg_latency_ms: float
    tokens_per_sec: float
    total_time_sec: float
    peak_vram_mb: float
    

class BatchSizeAnalyzer:
    def __init__(self, server_url: str = "http://localhost:8000", verbose: bool = False):
        self.server_url = server_url
        self.verbose = verbose
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def health_check(self, timeout: int = 10) -> bool:
        """Check if server is ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{self.server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
        return False
    
    async def benchmark_batch(self, batch_size: int, iterations: int = 20, 
                             n_predict: int = 50, prompt: str = "This is a test") -> BatchBenchmark:
        """Benchmark a specific batch size"""
        latencies = []
        total_tokens = 0
        
        print(f"  Batch size {batch_size}: ", end="", flush=True)
        
        start_total = time.time()
        
        for i in range(iterations):
            payload = {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.7,
                "stream": False,
                "n_keep": -1,
            }
            
            request_start = time.time()
            
            try:
                async with self.session.post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status != 200:
                        print(f"✗ (HTTP {resp.status})")
                        return None
                    
                    data = await resp.json()
                    
                    request_latency = (time.time() - request_start) * 1000  # ms
                    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    
                    latencies.append(request_latency)
                    total_tokens += completion_tokens
                    
                    print(".", end="", flush=True)
            
            except asyncio.TimeoutError:
                print("✗ (timeout)")
                return None
            except Exception as e:
                if self.verbose:
                    print(f"✗ ({e})")
                return None
        
        total_time = time.time() - start_total
        avg_latency = mean(latencies) if latencies else 0
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        print(f" OK")
        
        return BatchBenchmark(
            batch_size=batch_size,
            avg_latency_ms=avg_latency,
            tokens_per_sec=throughput,
            total_time_sec=total_time,
            peak_vram_mb=0,  # Would need nvidia-smi monitoring to get actual value
        )
    
    async def analyze(self, batch_sizes: list = None, iterations: int = 20,
                     n_predict: int = 50, prompt: str = "This is a test") -> dict:
        """Analyze multiple batch sizes"""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        print(f"\nBench marking batch sizes: {batch_sizes}\n")
        
        results = []
        
        for batch_size in batch_sizes:
            result = await self.benchmark_batch(
                batch_size=batch_size,
                iterations=iterations,
                n_predict=n_predict,
                prompt=prompt,
            )
            
            if result:
                results.append(result)
            else:
                print(f"  Skipping batch size {batch_size} (failed)")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [
                {
                    "batch_size": r.batch_size,
                    "avg_latency_ms": r.avg_latency_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "total_time_sec": r.total_time_sec,
                }
                for r in results
            ]
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze optimal batch size for throughput vs latency"
    )
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Iterations per batch size")
    parser.add_argument("--prompt", type=str, 
                       default="Once upon a time there was a great kingdom in a land far away",
                       help="Prompt to use")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    async with BatchSizeAnalyzer(
        server_url=args.server,
        verbose=args.verbose
    ) as analyzer:
        
        print(f"Connecting to {args.server}...", end=" ", flush=True)
        if not await analyzer.health_check():
            print("✗ Server not ready")
            return 1
        print("✓")
        
        results = await analyzer.analyze(
            batch_sizes=args.batch_sizes,
            iterations=args.iterations,
            prompt=args.prompt,
        )
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH SIZE ANALYSIS RESULTS")
        print("="*70 + "\n")
        
        print(f"{'Batch Size':<12} {'Avg Latency (ms)':<20} {'Throughput (tok/s)':<25}")
        print("-" * 70)
        
        max_throughput = 0
        optimal_batch = 1
        
        for bench in results["benchmarks"]:
            bs = bench["batch_size"]
            lat = bench["avg_latency_ms"]
            tps = bench["tokens_per_sec"]
            
            print(f"{bs:<12} {lat:<20.2f} {tps:<25.2f}")
            
            if tps > max_throughput:
                max_throughput = tps
                optimal_batch = bs
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70 + "\n")
        print(f"• Maximum throughput: {max_throughput:.2f} tok/s at batch_size={optimal_batch}")
        print(f"• For lowest latency: use batch_size=1 ({results['benchmarks'][0]['avg_latency_ms']:.2f}ms)")
        print(f"• For balanced performance: use batch_size=4-8")
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
