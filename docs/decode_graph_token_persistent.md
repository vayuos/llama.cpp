# Token-Persistent Decode Graph (Hard Invariant)

To eliminate CPU pacing and ensure GPU-owned continuous decode progression, the following hard invariants are now required and enforced by the runtime when CUDA graphs are enabled.

- **Token-persistent graph definition**: A single decode graph instance is created once before decode begins and reused for every token. The graph instance is token-persistent: it must not be torn down, rebuilt, or re-initialized between tokens.
- **Single execution handle**: Decode uses one persistent executable graph instance (`cudaGraphExec_t`) bound to the decode lifetime. Per-token graph submission, rebinding, or re-instantiation is prohibited.
- **Graph lifetime bound to decode lifetime**: The graph lifetime begins when decode mode is entered and ends when decode mode exits. Destroying or replacing the graph while decode mode is active is a hard error.
- **Token progression in GPU state**: Token index, KV offsets, and context indices are stored in graph-resident state visible to GPU kernels. The CPU MUST NOT update token position metadata per token.
- **No CPU token-loop orchestration**: During token generation the CPU only issues a single trigger to advance decode. The CPU must not advance token counters, mutate graph inputs per token, or patch parameters each iteration.
- **Stable graph inputs/placement**: Graph inputs have fixed shapes and locations for the duration of decode; only graph-internal state may change across tokens.
- **No per-token rebinding**: Attempts to rebind buffers, reassign backends, or re-evaluate tensor placement while in decode mode will fail fast.
- **GPU-owned progression**: The GPU is responsible for forward pass, KV cache updates, logits production, and advancing decode state. CPU-side participation in these steps is disallowed in decode mode.
- **Decode-mode guard**: Entering decode mode locks graph lifetime, backend selection, and execution plan. Any attempt to mutate them during decode aborts execution.
- **Invariant checks**: At runtime the system asserts that the graph pointer, execution plan, and backend assignment remain unchanged across decode steps; violations are fatal.

Usage note: The runtime exposes enter/exit APIs for decode mode. Callers must prepare and instantiate the decode graph once, call the `enter decode` API, then trigger GPU advances repeatedly without further host-side graph orchestration. When the decode sequence completes, call the `exit decode` API to allow graph teardown.
