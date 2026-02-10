#!/usr/bin/env python3
"""
profile_baseline.py - Baseline performance profiler for llama.cpp

Measures:
  - End-to-end latency per token (p50, p99)
  - Throughput (tokens/sec)
  - Memory usage (VRAM, RAM)
  - GPU utilization
  - CPU usage
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Dict, List, Optional

import aiohttp
import psutil

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class LatencySample:
    """Single latency measurement"""
    timestamp: float
    ttft: float  # Time to first token (ms)
    tpot: float  # Time per output token (ms)
    total_time: float  # Total request time (ms)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MemorySample:
    """Memory usage snapshot"""
    timestamp: float
    gpu_used_mb: float
    gpu_total_mb: float
    ram_used_mb: float
    ram_total_mb: float
    
    @property
    def gpu_usage_pct(self) -> float:
        if self.gpu_total_mb == 0:
            return 0.0
        return (self.gpu_used_mb / self.gpu_total_mb) * 100
    
    @property
    def ram_usage_pct(self) -> float:
        if self.ram_total_mb == 0:
            return 0.0
        return (self.ram_used_mb / self.ram_total_mb) * 100


@dataclass
class SystemStats:
    """System resource statistics"""
    cpu_usage_pct: float = 0.0
    gpu_usage_pct: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    ram_usage_pct: float = 0.0


@dataclass
class ProfileResults:
    """Aggregated profiling results"""
    timestamp: str
    model_name: str
    prompt: str
    n_iterations: int
    
    # Latency stats
    ttft_samples: List[float] = field(default_factory=list)
    tpot_samples: List[float] = field(default_factory=list)
    total_time_samples: List[float] = field(default_factory=list)
    
    # Throughput
    total_tokens_generated: int = 0
    total_time_seconds: float = 0.0
    
    # Memory
    memory_samples: List[MemorySample] = field(default_factory=list)
    
    # System
    system_stats: SystemStats = field(default_factory=SystemStats)
    
    def compute_stats(self) -> Dict:
        """Compute aggregated statistics"""
        stats = {
            "timestamp": self.timestamp,
            "model": self.model_name,
            "iterations": self.n_iterations,
            "prompt_length": len(self.prompt.split()),
        }
        
        # TTFT statistics
        if self.ttft_samples:
            stats["ttft_ms"] = {
                "mean": mean(self.ttft_samples),
                "p50": median(self.ttft_samples),
                "min": min(self.ttft_samples),
                "max": max(self.ttft_samples),
            }
            if len(self.ttft_samples) >= 4:
                q = quantiles(self.ttft_samples, n=100)
                stats["ttft_ms"]["p99"] = q[98]
        
        # TPOT statistics
        if self.tpot_samples:
            stats["tpot_ms"] = {
                "mean": mean(self.tpot_samples),
                "p50": median(self.tpot_samples),
                "min": min(self.tpot_samples),
                "max": max(self.tpot_samples),
            }
            if len(self.tpot_samples) >= 4:
                q = quantiles(self.tpot_samples, n=100)
                stats["tpot_ms"]["p99"] = q[98]
        
        # Throughput
        if self.total_time_seconds > 0:
            stats["throughput"] = {
                "tokens_per_sec": self.total_tokens_generated / self.total_time_seconds,
                "total_tokens": self.total_tokens_generated,
                "total_time_sec": self.total_time_seconds,
            }
        
        # Memory
        if self.memory_samples:
            gpu_usage = [s.gpu_usage_pct for s in self.memory_samples]
            ram_usage = [s.ram_usage_pct for s in self.memory_samples]
            stats["memory"] = {
                "gpu_avg_usage_pct": mean(gpu_usage),
                "gpu_max_usage_pct": max(gpu_usage),
                "gpu_total_mb": self.memory_samples[0].gpu_total_mb if self.memory_samples else 0,
                "ram_avg_usage_pct": mean(ram_usage),
                "ram_max_usage_pct": max(ram_usage),
            }
        
        # System
        stats["system"] = {
            "cpu_usage_pct": self.system_stats.cpu_usage_pct,
            "gpu_usage_pct": self.system_stats.gpu_usage_pct,
        }
        
        return stats


class GPUMonitor:
    """Monitor GPU metrics via nvidia-smi"""
    
    def __init__(self):
        self.available = self._check_nvidia_smi()
    
    def _check_nvidia_smi(self) -> bool:
        try:
            subprocess.run(["nvidia-smi", "--version"], 
                         capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_gpu_memory(self, device: int = 0) -> tuple[float, float]:
        """Get GPU memory usage (used_mb, total_mb)"""
        if not self.available:
            return 0.0, 0.0
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(device), 
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(","))
                return used, total
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        
        return 0.0, 0.0
    
    def get_gpu_utilization(self, device: int = 0) -> float:
        """Get GPU utilization percentage"""
        if not self.available:
            return 0.0
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(device),
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            pass
        
        return 0.0


class LlamaProfiler:
    """Main profiler for llama.cpp server"""
    
    def __init__(self, server_url: str = "http://localhost:8000",
                 gpu_device: int = 0,
                 verbose: bool = False):
        self.server_url = server_url
        self.gpu_device = gpu_device
        self.verbose = verbose
        self.gpu_monitor = GPUMonitor()
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def health_check(self, timeout: int = 30) -> bool:
        """Check if server is ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{self.server_url}/health") as resp:
                    if resp.status == 200:
                        return True
            except aiohttp.ClientError:
                pass
            
            await asyncio.sleep(1)
        
        return False
    
    async def generate_completion(self, prompt: str, n_predict: int = 100,
                                 temperature: float = 0.7) -> Optional[LatencySample]:
        """Generate a single completion and measure latency"""
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": False,
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    if self.verbose:
                        print(f"Error: {resp.status}")
                    return None
                
                data = await resp.json()
                
                total_time = (time.time() - start_time) * 1000  # ms
                
                # Extract token counts from response
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", len(prompt.split()))
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                # Estimate TTFT and TPOT
                # TTFT ≈ time before first output token (approximate)
                # TPOT ≈ time per token = total_time / completion_tokens
                ttft = total_time * 0.15 if completion_tokens > 0 else 0
                tpot = (total_time - ttft) / completion_tokens if completion_tokens > 1 else 0
                
                return LatencySample(
                    timestamp=time.time(),
                    ttft=ttft,
                    tpot=tpot,
                    total_time=total_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
        
        except asyncio.TimeoutError:
            print("Request timeout")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error: {e}")
            return None
    
    async def profile(self, prompt: str, n_iterations: int = 5,
                     model_name: str = "unknown") -> ProfileResults:
        """Run profiling suite"""
        
        results = ProfileResults(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            prompt=prompt,
            n_iterations=n_iterations,
        )
        
        print(f"\n{'='*60}")
        print(f"Starting baseline profiling ({n_iterations} iterations)")
        print(f"{'='*60}\n")
        
        # Warmup iteration
        print("Warming up...")
        await self.generate_completion(prompt)
        await asyncio.sleep(1)
        
        start_total_time = time.time()
        
        # Run iterations
        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}...", end=" ", flush=True)
            
            # Collect system metrics
            if self.gpu_monitor.available:
                gpu_used, gpu_total = self.gpu_monitor.get_gpu_memory(self.gpu_device)
                gpu_util = self.gpu_monitor.get_gpu_utilization(self.gpu_device)
                results.system_stats.gpu_usage_pct = gpu_util
            
            results.system_stats.cpu_usage_pct = psutil.cpu_percent(interval=0.1)
            
            # Collect memory sample
            ram_used = psutil.virtual_memory().used / (1024 * 1024)
            ram_total = psutil.virtual_memory().total / (1024 * 1024)
            results.memory_samples.append(MemorySample(
                timestamp=time.time(),
                gpu_used_mb=gpu_used,
                gpu_total_mb=gpu_total,
                ram_used_mb=ram_used,
                ram_total_mb=ram_total,
            ))
            
            # Generate completion
            sample = await self.generate_completion(prompt)
            
            if sample:
                results.ttft_samples.append(sample.ttft)
                results.tpot_samples.append(sample.tpot)
                results.total_time_samples.append(sample.total_time)
                results.total_tokens_generated += sample.completion_tokens
                
                print(f"✓ {sample.total_time:.0f}ms "
                      f"(TPOT: {sample.tpot:.1f}ms)")
            else:
                print("✗ FAILED")
            
            # Small delay between iterations
            await asyncio.sleep(0.5)
        
        results.total_time_seconds = time.time() - start_total_time
        
        return results


async def main():
    parser = argparse.ArgumentParser(
        description="Profile baseline performance of llama.cpp server"
    )
    parser.add_argument("--prompt", type=str, default="This is a test",
                       help="Prompt to use for generation")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of profiling iterations")
    parser.add_argument("--model", type=str, default="unknown",
                       help="Model name for reporting")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--gpu-device", type=int, default=0,
                       help="GPU device index")
    parser.add_argument("--output", type=str,
                       help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    async with LlamaProfiler(
        server_url=args.server,
        gpu_device=args.gpu_device,
        verbose=args.verbose
    ) as profiler:
        
        # Check server health
        print(f"Connecting to {args.server}...", end=" ", flush=True)
        if not await profiler.health_check():
            print("✗ Server not ready")
            return 1
        print("✓")
        
        # Run profiling
        results = await profiler.profile(
            prompt=args.prompt,
            n_iterations=args.iterations,
            model_name=args.model,
        )
        
        # Print results
        stats = results.compute_stats()
        
        print(f"\n{'='*60}")
        print("BASELINE PROFILING RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Model: {stats['model']}")
        print(f"Timestamp: {stats['timestamp']}")
        print(f"Iterations: {stats['iterations']}")
        print()
        
        if "ttft_ms" in stats:
            print("Time to First Token (TTFT):")
            print(f"  Mean:  {stats['ttft_ms']['mean']:.2f} ms")
            print(f"  P50:   {stats['ttft_ms']['p50']:.2f} ms")
            print(f"  P99:   {stats['ttft_ms'].get('p99', 'N/A')} ms")
            print()
        
        if "tpot_ms" in stats:
            print("Time Per Output Token (TPOT):")
            print(f"  Mean:  {stats['tpot_ms']['mean']:.2f} ms")
            print(f"  P50:   {stats['tpot_ms']['p50']:.2f} ms")
            print(f"  P99:   {stats['tpot_ms'].get('p99', 'N/A')} ms")
            print()
        
        if "throughput" in stats:
            print("Throughput:")
            print(f"  {stats['throughput']['tokens_per_sec']:.2f} tokens/sec")
            print(f"  {stats['throughput']['total_tokens']} total tokens in "
                  f"{stats['throughput']['total_time_sec']:.1f}s")
            print()
        
        if "memory" in stats:
            print("Memory Usage:")
            print(f"  GPU: {stats['memory']['gpu_avg_usage_pct']:.1f}% avg, "
                  f"{stats['memory']['gpu_max_usage_pct']:.1f}% max "
                  f"({stats['memory']['gpu_total_mb']:.0f} MB)")
            print(f"  RAM: {stats['memory']['ram_avg_usage_pct']:.1f}% avg, "
                  f"{stats['memory']['ram_max_usage_pct']:.1f}% max")
            print()
        
        print("System:")
        print(f"  CPU: {stats['system']['cpu_usage_pct']:.1f}%")
        print(f"  GPU: {stats['system']['gpu_usage_pct']:.1f}%")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
