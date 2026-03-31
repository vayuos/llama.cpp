---
description: Integration workflow between Gravitas Code Extension and Llama.cpp Server via Unix Domain Sockets
---

# Gravitas Code ↔ Llama.cpp Integration Workflow

This workflow automatically provisions a zero-latency environment between the Gravitas VS Code Extension and the `llama-server`. It replaces the standard TCP architecture `localhost:8080/8089` (used for older isolated setups) with direct inter-process communication via Unix Domain Sockets (UDS).

## 1. Socket Path Initialization
Instead of configuring TCP ports, the Gravitas Extension generates unique socket paths dynamically in the user's home directory.

**Gravitas Actions:**
- Detect OS (Linux/macOS) explicitly.
- Generate sockets `~/.gravitas/sockets/coder.sock` and `~/.gravitas/sockets/reviewer.sock`.
- Pass these absolute paths natively to the `llama-server` binary using `--host /path/to/coder.sock --port 0`.

## 2. Server Spawning & State Binding
The `llama-server` native to the repository (in `/home/viren/llama/llama.cpp/tools/server`) interprets the `.sock` suffix.

**Llama.cpp Actions:**
- The HTTP layer (`server-http.cpp`) recognizes the socket.
- It bypasses TCP initialization, configuring address family `AF_UNIX`.
- Sockets are created and bound. `llama-server` is instantly secured, preventing external network access directly to model endpoints.

## 3. Extension Network Dispatching (Axios)
With the Process running, Gravitas orchestrates inferences strictly through socket bindings.

**Axios Networking:**
- The HTTP Client in `GravitasCode` dynamically proxies its request.
- `baseURL` is set relative, and the `socketPath` property is embedded into Node.js's HTTPS/HTTP requests.
- Latency drops effectively to zero. Heavy context (like an entire file diff) transfers instantly without overhead.

## 4. Real-Rime Telemetry Streaming
The Gravitas UI monitors hardware health to protect and manage resources without running separate processes.

**Telemetry Loop:**
1. Gravitas Polls `/v1/hardware` endpoints or extended `/metrics` every `N` seconds over the socket.
2. `server.cpp` (Llama) dynamically calculates `GGML_BACKEND` memory footprints (VRAM vs CPU metrics) + slot allocations.
3. The JSON object stream translates inside the Webview to dynamic usage bars (`VRAM: 85%`).

## 5. Lifecycle Management and Clean Exit
When VS Code is closed, or "Stop Server" is clicked:
1. Gravitas sends a localized signal (`SIGTERM`) to the child `llama-server`.
2. `server.cpp` triggers clean up via memory-allocator sweeps and the `llama_backend_free`.
3. Node process safely unlinks the `.sock` files in `~/.gravitas/sockets/`.
