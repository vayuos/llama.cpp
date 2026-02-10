import json
import time
import requests
import numpy as np

def measure_latency(url, prompt, n_predict=128):
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": False
    }

    latencies = []
    
    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
            last_time = time.time()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        current_time = time.time()
                        chunk_data = json.loads(line[6:])
                        
                        # Skip if it's the final stop chunk without content
                        if chunk_data.get("stop", False) and not chunk_data.get("content", ""):
                            continue
                            
                        # Use the first token to mark the start of decoding
                        if len(latencies) == 0:
                            latencies.append(-1.0) # Mark first token
                        else:
                            latencies.append((current_time - last_time) * 1000) # ms
                        
                        last_time = current_time
    except Exception as e:
        print(f"Error during request: {e}")
        return None
                
    # Remove the first token placeholder
    latencies = [l for l in latencies if l > 0]
    
    if not latencies:
        return None
        
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    return {
        "avg": avg,
        "p50": p50,
        "p99": p99,
        "count": len(latencies),
        "raw": latencies
    }

if __name__ == "__main__":
    URL = "http://localhost:8080/completion"
    PROMPT = "Explain the importance of low latency in LLM serving."
    
    print(f"Benchmarking {URL}...")
    results = measure_latency(URL, PROMPT, n_predict=256)
    
    if results:
        print(f"Average Decode Latency: {results['avg']:.2f} ms")
        print(f"p50 Decode Latency:      {results['p50']:.2f} ms")
        print(f"p99 Decode Latency:      {results['p99']:.2f} ms")
        print(f"Tokens Generated:        {results['count']}")
    else:
        print("Failed to collect latency data.")
