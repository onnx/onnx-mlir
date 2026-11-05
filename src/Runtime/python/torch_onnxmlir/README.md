This package provides an onnxmlir-based compiler backend for torch.compile().

## Usage
Let's start with a simple torch model:
```python
import torch
import torch.nn as nn

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y  # Element-wise addition

mod = AddModel()

# Compile the model.
opt_mod = torch.compile(mod)

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

With torch_onnxmlir package, `torch.compile()` can be rewritten as follows:
```python
import torch
import torch.nn as nn

# Import torch_onnxmlir to use onnxmlir backend.
import torch_onnxmlir

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y  # Element-wise addition

mod = AddModel()

# Compile the model using onnxmlir backend in the torch_onnxmlir package.
om_option = {
    "compile_options": "-O3",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}
opt_mod = torch.compile(mod, backend="onnxmlir", options=om_options)

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

For more information about `torch.compile`, see its [document](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Caching the exported models and compiled libraries

To avoid recompiling models, the backend caches compiled models in the folder `${HOME}/.cache/torch_onnxmlir`.

Users can change the cache folder in two ways:

1. **Using environment variable**:
   ```bash
   export TORCHONNXMLIR_CACHE_DIR=/path/to/cache_folder
   ```

2. **Using config in Python**:
   ```python
   import torch_onnxmlir
   torch_onnxmlir.config.cache_dir = "/path/to/cache_folder"
   ```

### Security note on the disk cache

The cache directory is created with permissions `0o700` (owner read/write/execute only) so other local users cannot read or tamper with compiled artifacts.

Each cached `.so` (and `.constants.bin`, if present) has a SHA-256 hash stored alongside it at compile time. On reload, `load_from_disk` recomputes the hash before calling `dlopen` and treats any mismatch as a cache miss — the model is recompiled rather than the tampered file being loaded.

**The cache directory must not be shared with other local users or processes.** If you redirect the cache via `TORCHONNXMLIR_CACHE_DIR` or `torch_onnxmlir.config.cache_dir`, ensure the target path is private to the account running inference. Pointing multiple workers at a shared, world-writable location bypasses the hash check's protection by allowing an attacker to replace the `.so` and update the stored hash atomically.

Existing cache entries written by a version of `torch_onnxmlir` that predates this integrity check carry no stored hash. Those entries are treated as cache misses and recompiled automatically on the next run.

You can view the current cache directory using `explain_context()`:
```python
import torch_onnxmlir
with torch_onnxmlir.explain_context() as ctx:
    output = compiled_model(input)
    print(ctx)  # Shows "Cache Directory: ..." in output
```

To clean the cache and save disk space:
```bash
rm -rf ~/.cache/torch_onnxmlir
```

## Performance Analysis with explain()

The `explain()` feature provides insights into compilation and inference performance, helping you understand cache efficiency and identify bottlenecks.

### Basic Usage

Use the context manager for automatic metrics collection:

```python
import torch_onnxmlir

# Basic usage - metrics collected automatically
with torch_onnxmlir.explain_context() as metrics:
    output = compiled_model(input)
    print(metrics)  # Print formatted metrics

# Access metrics data programmatically
with torch_onnxmlir.explain_context(format="dict") as metrics:
    output = compiled_model(input)
    data = metrics.data
    print(f"Cache hits: {data['summary']['total_cache_hits']}")
    print(f"Total inferences: {data['summary']['total_inference_calls']}")

# Table format with custom sorting
with torch_onnxmlir.explain_context(format="table", sort_by="calls") as metrics:
    output = compiled_model(input)
    print(metrics)  # Prints formatted table

# Automatic export to file (simplifies logging)
with torch_onnxmlir.explain_context(format="json", export_to="metrics.json") as metrics:
    output = compiled_model(input)
    # Metrics automatically saved to metrics.json on exit

# Export with nested directories (auto-created)
with torch_onnxmlir.explain_context(export_to="logs/experiment1/metrics.json") as metrics:
    output = compiled_model(input)
    # Creates logs/experiment1/ directory if it doesn't exist
```

For per-op profiling (time spent in each ONNX op), see the
[Per-Op Profiling](#per-op-profiling) section below.

### Example Output

```
================================================================================
TORCH_ONNXMLIR PERFORMANCE METRICS
================================================================================

Unique Graphs: 1
Total Inference Calls: 10
Cache Hit Rate: 9/10 (90.0%)
Eager Fallbacks: 0

Per-Graph Statistics (sorted by Total Time):
--------------------------------------------------------------------------------
+------------------+-------+------+------------+-------------------+----------+-------+
| Graph ID         | Calls | Hits | Compile(s) | Avg Inference(ms) | Total(s) | %     |
+==================+=======+======+============+===================+==========+=======+
| model_hash_abc...| 10    | 9    | 1.234      | 5.20              | 1.286    | 100%  |
+------------------+-------+------+------------+-------------------+----------+-------+
```

## Per-Op Profiling

The `explain_context()` context manager can also collect per-op profiling data
from the onnx-mlir runtime, showing how much time each ONNX (or ZHigh) op
spends during inference.

### Enabling profiling

Pass `profiling=True` or a configuration dict:

```python
import torch_onnxmlir

# Simple: use defaults from config.
with torch_onnxmlir.explain_context(profiling=True) as ctx:
    for _ in range(5):
        output = compiled_model(input)
    print(ctx)

# Full control via dict.
with torch_onnxmlir.explain_context(
    profiling={"detail": 2, "warmup": 1, "reporting": "all"},
    format="dict",
) as ctx:
    for _ in range(5):
        output = compiled_model(input)
    print(ctx.data["profiling"])
```

### Profiling configuration keys

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `detail` | 0, 1, 2 | 0 | Detail level (see below). |
| `warmup` | int >= 0 | 0 | Number of initial runs to skip before collecting. |
| `reporting` | `"all"`, `"quartile"`, `"sum"` | `"all"` | How per-node times are aggregated across runs. |

Defaults can also be set globally via `torch_onnxmlir.config`:

```python
torch_onnxmlir.config.profiling_detail = 0
torch_onnxmlir.config.profiling_warmup = 0
torch_onnxmlir.config.profiling_reporting = "all"
```

### Detail levels

- **Level 0** (default) -- Op summary table only.
- **Level 1** -- Adds a tree showing tensor-shape signatures under each op,
  with local percentages relative to the parent op.
- **Level 2** -- Adds individual node names under each signature group, with
  local percentages relative to the parent signature.

Level 0 example (`profiling=True`, default):

```
Reporting: all -- average per-node time across all runs (5 measurement(s), 0 warmup skipped)

  Op              Count     Avg(ms)     Total(ms)        %
  ===========================================================
  onnx.Conv           2      0.0415        0.0830    72.2%
  ---------------------------------------------------------
  onnx.Gemm           1      0.0200        0.0200    17.4%
  ---------------------------------------------------------
  onnx.Relu           1      0.0120        0.0120    10.4%
  ---------------------------------------------------------

Total model time: 0.1150 ms
```

Level 1 example (`profiling={"detail": 1}`):

```
onnx.Conv (count: 3, avg: 0.0167, total: 0.0500, 80.0%)
├── 1x3x32x32xf32, 16x3x3x3xf32 (2, 0.0340, 68.0%)
└── 1x16x30x30xf32, 32x16x3x3xf32 (1, 0.0160, 32.0%)
```

Level 2 adds individual nodes:

```
onnx.Conv (count: 3, avg: 0.0167, total: 0.0500, 80.0%)
├── 1x3x32x32xf32, 16x3x3x3xf32 (2, 0.0340, 68.0%)
│   ├── /conv1 (0.0180, 52.9%)
│   └── /conv2 (0.0160, 47.1%)
└── 1x16x30x30xf32, 32x16x3x3xf32 (1, 0.0160, 32.0%)
    └── /conv3 (0.0160, 100.0%)
```

### Reporting modes

- **`"all"`** -- Average per-node time across all (non-warmup) runs. Count is
  the number of distinct node instances per op type. Best for steady-state
  latency measurement.
- **`"quartile"`** -- Sort runs, discard the top and bottom 25%, average the
  rest. Count is the number of distinct node instances. Useful when outlier
  runs (cold caches, GC pauses) skew the average.
- **`"sum"`** -- Cumulative time per node across all runs (not averaged). Count
  is the total number of invocations (distinct nodes x runs). Each (node, run)
  pair becomes a separate entry with that run's tensor-shape signature. This
  mode is designed for **decoding workloads** where tensor sizes grow by one
  each step -- the report shows every distinct shape seen across steps.

### Profiling in dict/json format

With `format="dict"` (or `"json"`), the profiling section is structured as:

```python
ctx.data["profiling"][cache_key] = {
    "total_time": ...,
    "num_measurements": ...,
    "reporting_mode": "all",
    "ops": [
        {
            "op_name": "onnx.Conv",
            "count": 3,
            "avg_time": 0.005,
            "total_time": 0.015,
            "percentage": 60.0,
            # detail >= 1:
            "signatures": [
                {
                    "signature": "1x3x32x32xf32, 16x3x3x3xf32",
                    "count": 2,
                    "avg_time": 0.006,
                    "total_time": 0.012,
                    "percentage": 80.0,   # local to op
                    # detail >= 2:
                    "nodes": [
                        {"node_name": "/conv1", "time": 0.007, "percentage": 58.3},
                        {"node_name": "/conv2", "time": 0.005, "percentage": 41.7},
                    ]
                },
            ]
        },
    ]
}
```

Percentages inside `signatures` are local to the parent op; percentages inside
`nodes` are local to the parent signature group.

### Sum mode for decoding

During autoregressive decoding, each generation step increases the sequence
length by one, so the same op node runs with a different tensor shape each
step. Using `reporting="sum"` captures this:

```python
with torch_onnxmlir.explain_context(
    profiling={"detail": 1, "warmup": 1, "reporting": "sum"},
) as ctx:
    with torch.no_grad():
        output = model.generate(**tokens, max_new_tokens=20)
    print(ctx)
```

The report will show every distinct shape that appeared across decoding steps,
grouped by op type, with cumulative times. For example, an `onnx.MatMul` node
that ran 20 times with shapes `1x1x128xf32`, `1x2x128xf32`, ...,
`1x20x128xf32` will have `count: 20` and 20 signature entries.

## Installation

The package torch_onnxmlir depends on the package `om_pyrt`. Follow the instruction [here](https://github.com/onnx/onnx-mlir/tree/main/src/Runtime/python/om_pyrt) to install `om_pyrt`. 

### Install from local directory
```bash
$ git clone --recursive https://github.com/onnx/onnx-mlir.git
$ cd onnx-mlir
$ pip install -e src/Runtime/python/torch_onnxmlir
```

### Install from pip repository
After the package is uploaded to pip server, you can install with `pip install torch_onnxmlir`.

## Run tests

By default, the tests use a local compiler specified in `utils.py`:
```
COMPILER_IMAGE_NAME = None
COMPILER_PATH = "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"
```

If using a remote compiler from a docker image, please change the two above variables.

### Run all tests

The folder `tests` contains many testcases to verify if the package works well or not.
These are steps to run tests:
```bash
$ cd tests
$ mkdir build
$ cd build
$ cmake ..
$ ctest -j 8
```

### Run a single test
- Use pytest to run a single test. For example,
```bash
$ python -m pytest test_add.py
```
