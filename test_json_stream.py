import urllib.request
import json
import time
import sys

def test_stream():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Count from 1 to 500."}],
        "stream": True
    }

    print(f"Connecting to {url}...")
    start = time.time()
    count = 0
    bytes_total = 0
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                if line:
                    bytes_total += len(line)
                    count += 1
                    # Optional: decode to verify it's valid JSON (sampled)
                    if count % 100 == 0:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data: ") and line_str != "data: [DONE]":
                            try:
                                json.loads(line_str[6:])
                            except json.JSONDecodeError:
                                print(f"Invalid JSON at line {count}: {line_str}")
    except Exception as e:
        print(f"Error: {e}")
        return

    end = time.time()
    duration = end - start
    print(f"Received {count} chunks, {bytes_total} bytes in {duration:.4f}s")
    if duration > 0:
        print(f"Speed: {count / duration:.2f} chunks/s")

if __name__ == "__main__":
    test_stream()
