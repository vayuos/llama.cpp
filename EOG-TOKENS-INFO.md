# EOG Tokens Configuration Guide

## What You're Seeing

At startup, you see warnings like:

```
load: control token: 151660 '<|fim_middle|>' is not marked as EOG
load: control token: 151659 '<|fim_prefix|>' is not marked as EOG
load: control token: 151653 '<|vision_end|>' is not marked as EOG
load: control token: 151648 '<|box_start|>' is not marked as EOG
```

Followed by:

```
load: printing all EOG tokens:
load:   - 151643 ('<|endoftext|>')
load:   - 151645 ('<|im_end|>')
load:   - 151662 ('<|fim_pad|>')
```

## What EOG Means

**EOG** = **End-Of-Generation** tokens

These are special tokens that signal "stop generating now":

```
Example conversation:
User: Hello
<generation starts>
Model: How can I help?
Model: <|im_end|>    ← EOG token, stops generation
<generation ends>
```

## The Issue Explained

Some control tokens are **not** marked as EOG:

| Token | Type | Is EOG? | Behavior |
|-------|------|---------|----------|
| `<|endoftext|>` | Chat terminator | ✅ Yes | Stops generation |
| `<|im_end|>` | Chat end marker | ✅ Yes | Stops generation |
| `<|fim_prefix|>` | Code FIM marker | ❌ No | Continues generation |
| `<|fim_middle|>` | Code FIM marker | ❌ No | Continues generation |
| `<|fim_suffix|>` | Code FIM marker | ❌ No | Continues generation |
| `<|vision_end|>` | Vision end | ❌ No | Continues generation |

**This is intentional** - FIM tokens are structural markers, not terminators.

## When This Matters

### Scenario 1: Chat Mode (Default) ✅
```
./llama-server -m model.gguf
# Generation stops at <|im_end|> automatically
# FIM tokens are never generated
# No impact
```

### Scenario 2: Fill-In-The-Middle Completion ⚠️
```
./llama-server -m model.gguf
# Using <|fim_prefix|> + <|fim_middle|> + <|fim_suffix|>
# Model generates code in middle section
# Expects <|fim_suffix|> to mark end of generated code
# But <|fim_suffix|> is NOT marked as EOG
# Generation may continue past the marker
```

### Scenario 3: Multimodal Vision ⚠️
```
./llama-server -m model.gguf
# Using vision tokens with <|vision_end|>
# Model generates text about image
# Expects <|vision_end|> to mark end
# But <|vision_end|> is NOT marked as EOG
# Generation may continue
```

## How to Handle

### Option 1: Ignore (For Chat Usage)

If you're doing standard chat, the warnings are harmless:

```bash
./llama-server -m model.gguf
# Warnings are just informational
# Chat mode works correctly
# Can safely ignore messages
```

### Option 2: Add Explicit Stop Sequences

For FIM or multimodal usage, explicitly stop:

```bash
./llama-server \
  -m model.gguf \
  --stop "<|fim_suffix|>" \
  --stop "<|vision_end|>"
```

**Server API call**:
```json
{
  "prompt": "<|fim_prefix|>def hello():\n<|fim_middle|>",
  "stop": ["<|fim_suffix|>", "\n\n"],
  "max_tokens": 256
}
```

### Option 3: Enforce Max Tokens

Always set a generation limit:

```bash
./llama-server -m model.gguf

# In API requests:
{
  "max_tokens": 256,  # Always limit generation
  "stop": ["<|fim_suffix|>"]
}
```

### Option 4: Modify GGUF (Advanced)

Mark additional tokens as EOG in GGUF metadata:

**Requires**: Re-exporting model with updated metadata
```python
# Pseudo-code
gguf_file = load_gguf("model.gguf")
eog_tokens = gguf_file["tokens.eog"]
eog_tokens.append(151660)  # Add <|fim_middle|>
gguf_file.save("model-updated.gguf")
```

**Note**: This is rarely necessary.

## For Different Use Cases

### Chat Only (Most Common) ✅
```bash
./llama-server -m model.gguf
# Just ignore warnings
# Everything works as expected
```

### Code Completion (FIM)
```bash
./llama-server -m model.gguf

# Request format:
{
  "prompt": "<|fim_prefix|>def hello():\n<|fim_middle|>",
  "stop": ["<|fim_suffix|>"],
  "max_tokens": 256
}

# Generates code between prefix and suffix
# Stops at suffix marker
```

### Vision Multimodal
```bash
./llama-server -m model.gguf

# Request format:
{
  "prompt": "<|vision_start|>[image]<|vision_end|>Describe this image",
  "stop": ["<|endoftext|>"],
  "max_tokens": 256
}

# Always include explicit stop
```

### Structured Output
```bash
./llama-server -m model.gguf

# Request format:
{
  "prompt": "JSON: {",
  "stop": ["}", "<|endoftext|>"],
  "max_tokens": 256
}

# Enforce stops with stop sequences
```

## Summary

| Use Case | Issue Impact | Recommended Action |
|----------|--------------|-------------------|
| Chat | None | Ignore warnings |
| Code (FIM) | May not stop at suffix | Add `--stop "<\|fim_suffix\|>"` |
| Vision | May not stop at end | Add explicit `stop` in requests |
| Structured | May not stop at markers | Use `max_tokens` + `stop` |

## Verification

To check if EOG is working correctly:

```bash
# Monitor generation in logs
./llama-server -m model.gguf -v  # Verbose mode

# Watch for:
# - Generation stops at EOG tokens
# - FIM tokens don't auto-stop (expected)
# - Explicit stop sequences work
```

## Conclusion

**The EOG warning is informational, not a bug.**

It simply tells you that certain control tokens (FIM, vision, box markers) are not configured as generation terminators.

This is intentional - these tokens are structural markers, not end-of-generation signals.

**Action required**: Only if you're using FIM or multimodal models - add explicit `--stop` sequences or use `max_tokens` limits.

**Default behavior**: Chat mode works perfectly without any changes.
