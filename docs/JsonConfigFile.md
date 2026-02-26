<!--- SPDX-License-Identifier: Apache-2.0 -->

# JSON Configuration File for onnx-mlir

## Overview

The `onnx-mlir` compiler supports loading configuration from a JSON file, allowing you to specify compile options and operation-specific settings in a single, reusable configuration file.

## Features

- **Compile Options**: Specify command-line options as an array in the config file
- **NNPA Operation Configs**: Configure device placement and quantization for specific operations
- **Default Config**: Automatically loads `omconfig.json` from the same directory as the input model if present
- **Custom Config**: Use `--config-file` to specify a custom configuration file
- **Config Override**: Config file options override command-line settings

## JSON Format

**Example:**
```json
{
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16"],
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "linear"
        },
        "rewrite": {
          "device": "NNPA",
          "quantize": true
        }
      }
    }
  ]
}
```

### Top-Level Keys

#### `compile_options` (array of strings, optional)
Array of compiler options that will be parsed as if they were passed on the command line.

**Example:**
```json
"compile_options": ["-O3", "-march=z16", "-mcpu=z16", "--enable-parallel"]
```

**Supported Options:**
- Optimization levels: `-O0`, `-O1`, `-O2`, `-O3`
- Target architecture: `-march=<arch>`, `-mcpu=<cpu>`, `-mtriple=<triple>`
- Accelerator: `-maccel=NNPA`
- Parallelization: `--enable-parallel`
- And all other onnx-mlir command-line options

#### `nnpa_ops_config` (array, optional)
Array of operation configuration rules for device placement and quantization when using NNPA accelerator. See [NNPA Configuration](#nnpa-configuration) below.

## Usage

### Using Default Config File

Create `omconfig.json` in the same directory as your input model:

```bash
cat > omconfig.json << 'EOF'
{
  "compile_options": ["-O3", "-march=z16"]
}
EOF

# Config is automatically loaded
onnx-mlir model.onnx
```

### Using Custom Config File

```bash
onnx-mlir --config-file=my-config.json model.onnx
```

### Config File Overrides CLI Options

Config file options take precedence over command-line settings:

```bash
# CLI has -O2, but config file overrides to -O3
onnx-mlir -O2 --config-file=config.json model.onnx
# Result: Uses -O3 from config file
```

## Examples

### Example 1: Optimization Only

```json
{
  "compile_options": ["-O3", "-march=native"]
}
```

### Example 2: NNPA Acceleration

```json
{
  "compile_options": ["-O3", "-march=z16", "-maccel=NNPA"],
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Conv"
        },
        "rewrite": {
          "device": "NNPA",
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul"
        },
        "rewrite": {
          "device": "NNPA"
        }
      }
    }
  ]
}
```

## NNPA Configuration
- [NNPA Accelerator Configuration](JsonConfigFile-NNPA.md)

## Troubleshooting

### Config File Not Found
If the config file doesn't exist, onnx-mlir continues without error (unless you explicitly specify `--config-file`).

### Parse Errors
If the JSON is malformed, you'll see an error message:
```
Warning: Failed to parse config file: config.json
```

### Invalid Options
If the config contains invalid compiler options, you'll see the standard option parsing error.

### Debugging
To see which options were loaded from the config, use `-v`:
```bash
onnx-mlir --config-file=config.json model.onnx -v
# Output contains the following information:
# Config file: config.json
# Onnx-mlir command: ./Debug/bin/onnx-mlir --config-file=config.json model.onnx -v -O2 -march=z16
```

## Related Documentation

- [NNPA Accelerator Configuration](JsonConfigFile-NNPA.md)
- [Compiler Options](Options.md)

## Migration from Command Line

If you have a complex command line:
```bash
onnx-mlir -O3 -march=z16 -mcpu=z16 --enable-parallel -j 8 model.onnx
```

Convert it to a config file:
```json
{
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16", "--enable-parallel", "-j 8"]
}
```

Then use:
```bash
onnx-mlir --config-file=config.json model.onnx
```
