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

You can view the current cache directory using the explain() feature:
```python
import torch_onnxmlir
torch_onnxmlir.config.enable_explain = True
# ... run your model ...
print(torch_onnxmlir.explain())  # Shows "Cache Directory: ..." in output
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
