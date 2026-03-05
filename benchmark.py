#!/usr/bin/env python3
"""
LLAMA.CPP Performance Benchmark using Python
No external dependencies required (uses built-in urllib)
"""

import urllib.request
import json
import time
import sys
from datetime import datetime

def make_request(host, port, prompt, max_tokens):
    """Make API request and return response with timing"""
    url = f"http://{host}:{port}/v1/chat/completions"

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    start_time = time.time()

    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        with urllib.request.urlopen(request, timeout=300) as response:
            response_data = json.loads(response.read().decode('utf-8'))

        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Extract metrics
        completion_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
        prompt_tokens = response_data.get('usage', {}).get('prompt_tokens', 0)

        return {
            'success': True,
            'elapsed_ms': elapsed,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'speed_tok_per_sec': (completion_tokens * 1000) / elapsed if elapsed > 0 else 0
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'elapsed_ms': (time.time() - start_time) * 1000
        }

def main():
    HOST = "127.0.0.1"
    PORT = 8080
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_FILE = f"/home/vayuos/llama/llama.cpp/benchmark_results_{TIMESTAMP}.txt"

    print("\n" + "="*50)
    print("LLAMA.CPP PYTHON BENCHMARK")
    print("="*50)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Results file: {METRICS_FILE}")
    print("")

    # Check if server is running
    print(f"Checking server at {HOST}:{PORT}...")
    test_request = make_request(HOST, PORT, "test", 1)
    if not test_request['success']:
        print(f"❌ ERROR: Cannot reach server at {HOST}:{PORT}")
        print(f"   Error: {test_request['error']}")
        print("")
        print("Make sure the server is running:")
        print(f"  /home/vayuos/llama/llama.cpp/build/bin/llama-server \\")
        print(f"    -m /home/vayuos/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \\")
        print(f"    -ngl 999 -c 8192 -b 4096 -ub 1024 --no-mmap --port 8080")
        sys.exit(1)

    print("✅ Server is ready!\n")

    # Test prompts
    tests = [
        ("SHORT PROMPT", "What is Python?", 128),
        ("MEDIUM PROMPT", "Explain how quicksort works. Include algorithm, complexity, and Python code.", 256),
        ("LONG PROMPT", "Design a REST API for task management using FastAPI. Include data models, CRUD endpoints, auth, error handling, and database integration.", 512)
    ]

    results = []

    for test_name, prompt, max_tokens in tests:
        print("="*50)
        print(f"TEST: {test_name}")
        print("="*50)

        result = make_request(HOST, PORT, prompt, max_tokens)
        results.append(result)

        if result['success']:
            print(f"Elapsed: {result['elapsed_ms']:.0f}ms")
            print(f"Tokens: {result['completion_tokens']} completion + {result['prompt_tokens']} prompt")
            print(f"Speed: {result['speed_tok_per_sec']:.2f} tok/sec")
            print("")
        else:
            print(f"❌ ERROR: {result['error']}")
            print("")

    # Write results to file
    with open(METRICS_FILE, 'w') as f:
        f.write("="*50 + "\n")
        f.write("BENCHMARK RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Server: {HOST}:{PORT}\n")
        f.write(f"Model: Qwen3-Coder-Next-UD-Q4_K_XL\n")
        f.write(f"Configuration: -ngl 999 -c 8192 -b 4096 -ub 1024 --no-mmap\n")
        f.write("\n")

        # Calculate average throughput
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_speed = sum(r['speed_tok_per_sec'] for r in successful_results) / len(successful_results)
            f.write(f"Average Throughput: {avg_speed:.2f} tok/sec\n")
            f.write("\n")

        # Details for each test
        for i, (test_name, _, _) in enumerate(tests):
            result = results[i]
            f.write(f"\n{test_name}:\n")
            if result['success']:
                f.write(f"  Elapsed: {result['elapsed_ms']:.0f}ms\n")
                f.write(f"  Completion Tokens: {result['completion_tokens']}\n")
                f.write(f"  Prompt Tokens: {result['prompt_tokens']}\n")
                f.write(f"  Speed: {result['speed_tok_per_sec']:.2f} tok/sec\n")
            else:
                f.write(f"  ERROR: {result['error']}\n")

        f.write("\n" + "="*50 + "\n")
        f.write("PERFORMANCE TARGETS\n")
        f.write("="*50 + "\n")
        f.write("Baseline (previous): 405 tok/sec\n")
        f.write("Expected (optimized): 475-560 tok/sec\n")
        f.write("Improvement: +17-38%\n")

        if successful_results:
            actual = sum(r['speed_tok_per_sec'] for r in successful_results) / len(successful_results)
            improvement = ((actual - 405) / 405) * 100
            f.write(f"\nACTUAL RESULT: {actual:.2f} tok/sec ({improvement:+.1f}%)\n")

    print("="*50)
    print("SUMMARY")
    print("="*50)

    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_speed = sum(r['speed_tok_per_sec'] for r in successful_results) / len(successful_results)
        improvement = ((avg_speed - 405) / 405) * 100

        print(f"✅ Average Throughput: {avg_speed:.2f} tok/sec")
        print(f"📊 Improvement vs baseline: {improvement:+.1f}%")

        if avg_speed >= 475:
            print("🎉 EXCELLENT - Meets/exceeds target!")
        elif avg_speed >= 450:
            print("✅ GOOD - Within acceptable range")
        else:
            print("⚠️  Below target - Check GPU utilization")
    else:
        print("❌ Benchmark failed - No successful results")

    print(f"\nFull results saved to: {METRICS_FILE}")
    print("\nTo view results:")
    print(f"  cat {METRICS_FILE}")

if __name__ == "__main__":
    main()
