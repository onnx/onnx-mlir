---
name: using-torch-onnxmlir-with-huggingface
description: 'Use when compiling and running PyTorch or HuggingFace Transformers models through torch.compile() with the torch_onnxmlir backend: installation, encoder/decoder usage, configuration options, compiled-model caching, and debugging.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/using-torch-onnxmlir-with-huggingface.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Using torch_onnxmlir with HuggingFace Models

Tutorial for using the `torch_onnxmlir` package to compile and run HuggingFace Transformers models via `torch.compile()` with onnx-mlir as the backend.

**Applies to:** `src/Runtime/python/torch_onnxmlir/`, `docs/RunTorchModel.md`

**Related pages:** Runtime, [Compiler](../../../.agents/skills/onnx-mlir-internals/references/compiler.md)

---

## Overview

`torch_onnxmlir` provides a custom backend for PyTorch's `torch.compile()` that routes compilation through onnx-mlir instead of the default TorchInductor backend. This enables:

- **Transparent acceleration**: Compile PyTorch models (especially HuggingFace Transformers) with onnx-mlir optimizations without changing model code
- **NNPA acceleration**: Target IBM z16 NNPA hardware acceleration via `-march=z16 -maccel=NNPA` compiler flags
- **Persistent caching**: Compiled models are cached on disk to avoid recompilation across runs
- **HuggingFace integration**: Works seamlessly with `transformers.AutoModel`, `AutoModelForCausalLM`, and related classes

The backend sits between PyTorch's FX graph representation and onnx-mlir's compilation pipeline:

```
torch.nn.Module
  → torch.compile(backend="onnxmlir")
  → torch FX graph
  → torch_onnxmlir backend (backend.py)
  → ONNX export
  → om_pyrt.CompileSession (onnx-mlir compilation)
  → om_pyrt.InferenceSession (execution)
  → cached .so library
```

## Installation

### Prerequisites

1. **onnx-mlir built and installed**: You need a working onnx-mlir build with Python bindings
2. **om_pyrt installed**: The `torch_onnxmlir` package depends on `om_pyrt` (see [om_pyrt README](https://github.com/onnx/onnx-mlir/tree/main/src/Runtime/python/om_pyrt))

### Install torch_onnxmlir

From local onnx-mlir repository:

```bash
cd onnx-mlir
pip install -e src/Runtime/python/torch_onnxmlir
```

Or from PyPI (when available):

```bash
pip install torch_onnxmlir
```

### Verify Installation

```python
import torch_onnxmlir
print(torch_onnxmlir.__file__)  # Should show installation path
```

## Basic Usage Pattern

### Minimal Example

```python
import torch
import torch.nn as nn
import torch_onnxmlir  # Registers "onnxmlir" backend

class AddModel(nn.Module):
    def forward(self, x, y):
        return x + y

model = AddModel()

# Compile with onnx-mlir backend
om_options = {
    "compile_options": "-O3",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}
compiled_model = torch.compile(model, backend="onnxmlir", options=om_options)

# Use normally
input1 = torch.randn(2)
input2 = torch.randn(2)
output = compiled_model(input1, input2)
```

### Backend Options

The `options` dict passed to `torch.compile()` supports:

- **`compiler_path`** (str): Path to `onnx-mlir` binary (e.g., `/workdir/onnx-mlir/build/Debug/bin/onnx-mlir`)
- **`compiler_image_name`** (str or None): Docker image name for containerized compilation (e.g., `"onnxmlirczar/onnx-mlir:latest"`). Set to `None` for local compilation.
- **`compile_options`** (str): Space-separated compiler flags passed to onnx-mlir (e.g., `"-O3 -march=z16 -maccel=NNPA"`)

Example with NNPA acceleration:

```python
om_options = {
    "compiler_image_name": None,
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    "compile_options": "-O3 -march=z16 -maccel=NNPA",
}
compiled_model = torch.compile(model, backend="onnxmlir", options=om_options)
```

## HuggingFace Encoder Models

Encoder models (BERT-style, embedding models) process input sequences and return contextualized representations. Common use cases: sentence embeddings, classification, named entity recognition.

### Example: Granite Embedding Model

```python
import torch
import torch_onnxmlir
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model_path = "ibm-granite/granite-embedding-30m-english"
model = AutoModel.from_pretrained(model_path, dtype="float32")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set to inference mode
model.eval()

# Compile with onnx-mlir backend
om_options = {
    "compiler_image_name": None,
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    "compile_options": "-O3 -march=z16 -maccel=NNPA",
}
compiled_model = torch.compile(model, backend="onnxmlir", options=om_options)

# Prepare inputs
text = "AI is fascinating."
inputs = tokenizer(text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = compiled_model(**inputs)

# Extract CLS token embedding
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(cls_embedding.shape)  # [1, hidden_size]
```

### Supported Encoder Models

Tested models (from `test_encoder_models.py`):

- `ibm-granite/granite-embedding-30m-english`
- `ibm-granite/granite-embedding-278m-multilingual`
- `google-bert/bert-base-uncased`

### Key Points for Encoders

1. **Set `dtype="float32"`**: Ensures compatibility with onnx-mlir's type system
2. **Use `model.eval()`**: Disables dropout and other training-specific behavior
3. **Wrap in `torch.no_grad()`**: Disables gradient computation for inference
4. **First run compiles**: The first inference call triggers compilation and caching; subsequent calls use the cached model
5. **Variable sequence lengths**: Different input lengths may trigger separate compilations (one per unique shape)

## HuggingFace Decoder Models

Decoder models (GPT-style, causal language models) generate text autoregressively. They require special handling for the prefill and decode phases.

### Example: GPT-2 Text Generation

```python
import torch
import torch_onnxmlir
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="eager",  # Required: decomposes SDPA ops
    dtype="float32",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set to inference mode
model.eval()

# Compile the forward method (not the whole model)
om_options = {
    "compiler_image_name": None,
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    "compile_options": "-O3 -march=z16 -maccel=NNPA",
}
model.forward = torch.compile(model.forward, backend="onnxmlir", options=om_options)

# Prepare inputs
prompt = "Hello, I'm a language model,"
input_tokens = tokenizer(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    output = model.generate(**input_tokens, max_new_tokens=11)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Supported Decoder Models

Tested models (from `test_decoder_models.py`):

- `openai-community/gpt2`
- `Qwen/Qwen2-0.5B-Instruct`
- `ibm-granite/granite-3.3-2b-instruct`
- `ibm-granite/granite-4.0-350M`

### Key Points for Decoders

1. **Use `attn_implementation="eager"`**: Required to decompose scaled dot-product attention (SDPA) ops into primitives that onnx-mlir can handle
2. **Compile `model.forward`, not `model`**: The `generate()` method orchestrates multiple forward passes; compile the forward method itself
3. **Two compilations occur**: One for the prefill phase (processing the prompt) and one for the decode phase (generating tokens one at a time with KV cache)
4. **KV cache handling**: The backend automatically manages key-value cache tensors across generation steps
5. **Use `torch.no_grad()`**: Essential for generation to avoid memory overhead

### Prefill vs. Decode Phases

Decoder models have two distinct execution patterns:

- **Prefill**: Process the entire prompt at once (shape: `[batch, prompt_length, ...]`)
- **Decode**: Generate one token at a time using cached keys/values (shape: `[batch, 1, ...]`)

The backend compiles these separately because they have different input shapes and computational patterns. You'll see two compilation messages in the logs on the first run.

## Configuration Options

The `torch_onnxmlir.config` module provides runtime configuration knobs:

```python
import torch_onnxmlir

# Custom cache directory (default: ~/.cache/torch_onnxmlir, changed from
# the bare ~/.cache in a 2026-07-29 update — see Caching Behavior below)
torch_onnxmlir.config.cache_dir = "/path/to/custom/cache"

# Maximum number of cached sessions (default: 3)
torch_onnxmlir.config.session_cache_limit = 5

# Keep ONNX files after compilation (default: False)
torch_onnxmlir.config.keep_onnx_files = True

# Generate test data sets for debugging (default: False)
torch_onnxmlir.config.generate_test_data_set = True

# Regenerate test data sets on every run (default: False)
torch_onnxmlir.config.regenerate_test_data_set = True
```

### Configuration Details

- **`cache_dir`**: Where compiled models are stored. Can also be set via `TORCHONNXMLIR_CACHE_DIR` environment variable.
- **`session_cache_limit`**: LRU cache size for in-memory sessions. Older sessions are evicted when the limit is reached.
- **`keep_onnx_files`**: Useful for debugging; keeps intermediate ONNX files alongside compiled `.so` libraries.
- **`generate_test_data_set`**: Creates `test_data_set/` folder with `input_*.pb` and `output_*.pb` files usable with `RunONNXModel.py` for standalone testing.
- **`regenerate_test_data_set`**: Overwrites test data on every run (slow; use only for debugging).

### Hashing and Cache Key Generation

The backend uses a sophisticated hashing mechanism to determine if a model needs recompilation:

- **Graph structure**: FX graph topology and operations, including each node's literal arguments (reshape dims, axis indices, scalar thresholds), so two graphs that differ only in a constant argument get different keys
- **Parameter values**: Each parameter is reduced to a SHA-256 of its shape, dtype and data. A parameter with at most 1024 elements is hashed in full; a larger one is hashed from 1024 evenly strided samples. Nested sub-module parameters (e.g. `linear.weight`) are included. `config.sample_parameter_values_limit` is still defined in `config.py` but the hash no longer reads it
- **Input shapes**: Each placeholder's shape and dtype are part of the cache key, with dynamic (`SymInt`) dims renamed to stable `dim_0`, `dim_1`, ...
- **Compiler options**: Different flags produce different cache entries

If the hash matches an existing cache entry, the compiled model is reused. Otherwise, a new compilation is triggered.

## Caching Behavior

### Cache Location

Default: `~/.cache/torch_onnxmlir/`

Override via:
```python
torch_onnxmlir.config.cache_dir = "/custom/path"
```

Or environment variable:
```bash
export TORCHONNXMLIR_CACHE_DIR=/custom/path
```

### Cache Structure

```
~/.cache/torch_onnxmlir/
├── <hash1>/
│   ├── model.so          # Compiled library
│   ├── om_backend_config.json  # SHA-256 of model.so (and .constants.bin), signatures, compile info
│   ├── model.onnx        # (if keep_onnx_files=True)
│   └── test_data_set/    # (if generate_test_data_set=True)
│       ├── input_0.pb
│       ├── output_0.pb
│       └── ...
├── <hash2>/
│   └── ...
└── ...
```

Each unique model (by hash) gets its own subdirectory. Directories are created with mode `0o700` (owner only).

### Cache Integrity

When a compiled model is cached, a SHA-256 hash of each `.so`/`.constants.bin` is written under `artifact_hashes` in that entry's `om_backend_config.json`. Before loading a cached `.so`, the backend recomputes the hashes. Any of these counts as a cache miss, and the model is recompiled instead of loaded:

- a hash does not match, or a hashed file is missing
- the entry has no stored hashes (written by a `torch_onnxmlir` version before this check), or no config file
- a hash key is not a bare file name

So the first run after upgrading recompiles every cached model once. Keep the cache directory private to the account running inference: anyone who can write to it can replace both the `.so` and its stored hash.

### Cache Invalidation

The cache is invalidated when:

- Model parameters change (weights updated)
- Graph structure changes (model architecture modified)
- Compiler options change (different flags)
- Input shapes change (for shape-dependent compilations)

### Session Cache (In-Memory)

In addition to disk caching, the backend maintains an in-memory LRU cache of `InferenceSession` objects (default limit: 3). This avoids reloading the `.so` library from disk on repeated calls with the same model.

## Testing and Debugging

### Running Tests

The `torch_onnxmlir` package includes comprehensive tests:

```bash
cd onnx-mlir/src/Runtime/python/torch_onnxmlir/tests
mkdir build && cd build
cmake ..
ctest -j 8
```

Or run individual tests with pytest:

```bash
python -m pytest test_add.py
python -m pytest test_encoder_models.py
python -m pytest test_decoder_models.py
```

### Test Coverage

- **`test_add.py`**: Simple element-wise addition (sanity check)
- **`test_gemm.py`**: Matrix multiplication
- **`test_relu.py`**: Activation function
- **`test_encoder_models.py`**: BERT/Granite embedding models
- **`test_decoder_models.py`**: GPT-2/Qwen2/Granite causal LM models
- **`test_sessioncache.py`**: Cache behavior verification
- **`test_cache_integrity.py`**: Disk-cache hash check and directory mode (no compiler needed)
- **`test_hash_key.py`**: Cache-key generation: literal arguments and parameter values change the key (no compiler needed; `conftest.py` stubs out `om_pyrt` when no onnx-mlir build is installed)

The tests use the compiler at `/workdir/onnx-mlir/build/Debug/bin/onnx-mlir` (the container path) unless the `COMPILER_PATH` environment variable names another `onnx-mlir` binary.

### Debugging Tips

1. **Enable logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Keep ONNX files**:
   ```python
   torch_onnxmlir.config.keep_onnx_files = True
   ```
   Inspect the exported ONNX graph in the cache directory.

3. **Generate test data sets**:
   ```python
   torch_onnxmlir.config.generate_test_data_set = True
   ```
   Use `RunONNXModel.py` to test the compiled model standalone:
   ```bash
   python utils/RunONNXModel.py --load-model ~/.cache/torch_onnxmlir/<hash>/model.so
   ```

4. **Check compiler output**:
   Add `-v` to `compile_options` for verbose onnx-mlir output:
   ```python
   om_options = {"compile_options": "-O3 -v"}
   ```

5. **Verify NNPA usage**:
   Add `--onnx-op-stats TXT` to see which ops are accelerated:
   ```python
   om_options = {"compile_options": "-O3 -march=z16 -maccel=NNPA --onnx-op-stats TXT"}
   ```

6. **Cache-hit/compile-time metrics with `explain_context()`** (added 2026-07-31, replacing an earlier one-shot `explain()`/`config.enable_explain` API from two days prior — some upstream README sections still show the older snippet):
   ```python
   with torch_onnxmlir.explain_context() as metrics:
       output = compiled_model(input)
       print(metrics)  # formatted table: unique graphs, cache hit rate,
                        # eager fallbacks, per-graph compile/inference time
   ```
   `format="dict"` gives programmatic access (`metrics.data["summary"]["total_cache_hits"]`, etc.); `sort_by="calls"` reorders the per-graph table; `export_to="path/to/file.json"` (nested dirs auto-created) saves it instead of/alongside printing. This is the fastest way to confirm whether a slow first call was compilation (expected) versus a cache miss that shouldn't have happened (e.g. from an input shape that varies more than expected).

### Common Issues

**Issue**: `attn_implementation="eager"` required for decoder models

**Reason**: PyTorch's fused SDPA op is not directly supported by ONNX export. Using `"eager"` decomposes it into primitive ops (matmul, softmax, etc.).

**Issue**: Different sequence lengths trigger recompilation

**Reason**: Input shapes are part of the cache key. Use padding to a fixed length if you want to avoid multiple compilations.

**Issue**: First run is slow

**Reason**: Compilation happens on the first inference call. Subsequent runs use the cached model.

## Comparison with utils/python/transformers Scripts

The `torch_onnxmlir` package and the scripts in `utils/python/transformers/` serve different purposes:

### torch_onnxmlir (This Tutorial)

- **Purpose**: Transparent PyTorch integration via `torch.compile()`
- **Interface**: Python, PyTorch-native
- **Use case**: Drop-in replacement for default `torch.compile()` backend
- **Model loading**: Uses `transformers.AutoModel` directly
- **Caching**: Automatic, hash-based
- **Best for**: Iterative development, experimentation, HuggingFace workflows

### utils/python/transformers Scripts

- **Purpose**: Standalone ONNX model execution (e.g., `run_gpt2_from_huggingface.py`)
- **Interface**: Python, but uses `PyOMCompile` and `OMExecutionSession` directly
- **Use case**: Pre-export ONNX models from HuggingFace, compile once, run many times
- **Model loading**: Downloads ONNX models from HuggingFace Hub (not PyTorch models)
- **Caching**: Manual (user manages downloaded ONNX files)
- **Best for**: Production deployments, benchmarking, quantization experiments

### When to Use Which

| Scenario | Recommended Approach |
|----------|---------------------|
| Prototyping with HuggingFace models | `torch_onnxmlir` |
| Integrating into existing PyTorch training/inference code | `torch_onnxmlir` |
| Benchmarking onnx-mlir performance | `utils/python/transformers` scripts |
| Production deployment (pre-compiled models) | `utils/python/transformers` scripts or direct `om_pyrt` usage |
| Experimenting with quantization | `utils/python/transformers` scripts (see `run_gpt2_from_huggingface.py` with `-march=z16 -maccel=NNPA`) |

Both approaches ultimately use the same onnx-mlir compiler and runtime; the difference is in the level of abstraction and integration with PyTorch.

## Advanced Topics

### Custom FX Graph Transformations

The backend applies several FX graph transformations before ONNX export (in `fx_utils.py`):

- **Freeze scalar constants**: Convert scalar Python values to tensor constants
- **Convert `SymInt` to tensors**: Handle dynamic shape symbols
- **Remove unused placeholders**: Clean up graph inputs that aren't actually used

These transformations ensure the exported ONNX graph is compatible with onnx-mlir's expectations.

### Session Cache Implementation

The `SessionCache` class (in `sessioncache.py`) implements an LRU cache with disk persistence:

- **Cache key**: `_om_` followed by three SHA-256 digests: the FX graph (node args and parameter hashes included), the sorted compiler options, and the example inputs' ranks
- **Cache value**: `CacheValue` object containing the compiled `.so` path and metadata
- **Eviction policy**: LRU (least recently used) when `session_cache_limit` is reached
- **Persistence**: Each entry's metadata (`artifact_hashes`, `example_inputs_indices`, `compilation_info`, `input_signature`, `output_signature`) is stored in `om_backend_config.json` inside that entry's directory; `load_from_disk` verifies the hashes before creating the `InferenceSession`

### Compiler Options Passthrough

The `compile_options` string is passed directly to onnx-mlir's CLI parser. All flags documented in `docs/Options.md` are available:

- Optimization levels: `-O0`, `-O1`, `-O2`, `-O3`
- Target architecture: `-march=x86-64`, `-march=z16`
- Accelerators: `-maccel=NNPA`
- Profiling: `--profile-ir=Onnx` (timing) or `--profile-ir-with-sig=Onnx` (timing + shapes); the raw `--instrument-*` flags these subsume are deprecated for everyday use
- Emission targets: `--EmitONNXIR`, `--EmitMLIR`, `--EmitLLVMIR` (for debugging)
- Op statistics: `--onnx-op-stats TXT`

See [Compiler Options](../../../.agents/skills/onnx-mlir-internals/references/compiler.md) for the full list.

## Summary

`torch_onnxmlir` provides a seamless way to leverage onnx-mlir's optimizations and NNPA acceleration for HuggingFace Transformers models via PyTorch's `torch.compile()` API. Key takeaways:

1. **Installation**: Requires `om_pyrt` and a working onnx-mlir build
2. **Basic usage**: `torch.compile(model, backend="onnxmlir", options={...})`
3. **Encoder models**: Work out-of-the-box with `AutoModel`
4. **Decoder models**: Require `attn_implementation="eager"` and compiling `model.forward`
5. **Caching**: Automatic, hash-based, with configurable cache directory and limits; cached libraries are hash-checked before loading
6. **Debugging**: Use `keep_onnx_files`, `generate_test_data_set`, and verbose compiler flags
7. **Comparison**: `torch_onnxmlir` for PyTorch integration, `utils/python/transformers` scripts for standalone ONNX execution

## Verification

This skill was grounded against the following sources:

**Repositories and SHAs:**
- `onnx-mlir@f4b8e8de`: Main source for torch_onnxmlir implementation and documentation
- `onnx-mlir@c92e126` (2026-08-06 sync): re-verified `cache_dir` default and `explain_context()` API against this later commit — see `sync-status.md`'s `94c6bc1`/`de23de78`/`7cea64f` entries for the commits that introduced/changed these
- `onnx-mlir@b505d64`: re-verified Hashing and Cache Key Generation, the cache-key bullet under Session Cache Implementation, and Test Coverage against `backend.py` (`generate_hash_key`, `_stable_args_str`, `_tensor_sha256`, `_PARAM_FULL_HASH_LIMIT = 1024`), `config.py`, `tests/test_hash_key.py` and `tests/conftest.py`; also re-checked the `config` defaults, `TORCHONNXMLIR_CACHE_DIR`, the tested encoder/decoder model lists and the `ENCODER_MODEL`/`DECODER_MODEL`/`COMPILER_PATH` test variables
- `onnx-mlir@5534bc7`: re-verified the disk-cache layout, `0o700` directory mode, and hash check (Cache Structure, Cache Integrity, Session Cache Implementation) against `sessioncache.py` and the README's "Security note on the disk cache"; `tests/test_cache_integrity.py` covers them

**Files Read:**
- `src/Runtime/python/torch_onnxmlir/README.md` — Installation, basic usage, caching, `explain_context()` usage/example output
- `src/Runtime/python/torch_onnxmlir/src/torch_onnxmlir/backend.py` — Backend implementation, FX graph handling, cache-key generation
- `src/Runtime/python/torch_onnxmlir/src/torch_onnxmlir/config.py` — Configuration options
- `src/Runtime/python/torch_onnxmlir/src/torch_onnxmlir/sessioncache.py` — Cache implementation
- `src/Runtime/python/torch_onnxmlir/src/torch_onnxmlir/explain.py` — `explain_context()`/`is_enabled()` implementation
- `src/Runtime/python/torch_onnxmlir/tests/test_encoder_models.py` — Encoder model examples
- `src/Runtime/python/torch_onnxmlir/tests/test_decoder_models.py` — Decoder model examples
- `src/Runtime/python/torch_onnxmlir/tests/test_hash_key.py`, `conftest.py` — Cache-key tests and the `om_pyrt` stub
- `utils/python/transformers/run_gpt2_from_huggingface.py` — Comparison approach

**How to Test:**

1. **Basic functionality test** (from README.md):
   ```bash
   python -c "
   import torch
   import torch.nn as nn
   import torch_onnxmlir
   
   class AddModel(nn.Module):
       def forward(self, x, y):
           return x + y
   
   model = AddModel()
   om_options = {
       'compile_options': '-O3',
       'compiler_path': '/workdir/onnx-mlir/build/Debug/bin/onnx-mlir',
   }
   compiled = torch.compile(model, backend='onnxmlir', options=om_options)
   
   x, y = torch.randn(2), torch.randn(2)
   result = compiled(x, y)
   print('Success:', result.shape)
   "
   ```

2. **Run test suite**:
   ```bash
   cd onnx-mlir/src/Runtime/python/torch_onnxmlir/tests
   python -m pytest test_add.py
   ```

3. **Encoder model test** (requires HuggingFace transformers):
   ```bash
   ENCODER_MODEL=bert-base-uncased python -m pytest test_encoder_models.py
   ```

4. **Decoder model test**:
   ```bash
   DECODER_MODEL=gpt2 python -m pytest test_decoder_models.py
   ```

Expected: All tests pass, models compile and run with accuracy >98% similarity to PyTorch reference.

For more details on the runtime architecture, see Runtime. For compiler flag reference, see [Compiler](../../../.agents/skills/onnx-mlir-internals/references/compiler.md).

---
