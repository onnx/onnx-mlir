# JSON Configuration File for onnx-mlir

## Overview

The `onnx-mlir` compiler supports loading configuration from a JSON file, allowing you to specify compile options and operation-specific settings in a single, reusable configuration file.

## Features

- **Compile Options**: Specify command-line options as a string in the config file
- **NNPA Operation Configs**: Configure device placement and quantization for specific operations
- **Default Config**: Automatically loads `omconfig.json` from the current directory if present
- **Custom Config**: Use `--config-file` to specify a custom configuration file
- **CLI Override**: Command-line options override config file settings

## JSON Format

```json
{
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16"],
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Conv",
          "onnx_node_name": "conv1"
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

Create `omconfig.json` in your working directory:

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

### Overriding Config with CLI

Command-line options take precedence over config file settings:

```bash
# Config has -O3, but CLI overrides to -O2
onnx-mlir --config-file=config.json -O2 model.onnx
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
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16", "-maccel=NNPA"],
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

### Example 3: Development Build

```json
{
  "compile_options": ["-O0", "--preserveMLIR", "--preserveLLVMIR", "--printIR"]
}
```

### Example 4: Production Build with Parallelization

```json
{
  "compile_options": ["-O3", "-march=z16", "--enable-parallel", "-j", "8"]
}
```

## NNPA Configuration

The `nnpa_ops_config` section allows fine-grained control over operation placement and quantization for NNPA accelerator.

### Match Criteria

- **`node_type`** (required): Operation type to match (supports regex)
  - Example: `"onnx.Conv"`, `"onnx.*"` (all ONNX ops)
- **`onnx_node_name`** (optional): Specific operation name (supports regex)
  - Example: `"conv1"`, `"conv.*"` (all ops starting with "conv")

### Rewrite Actions

- **`device`** (optional): Target device for the operation
  - Values: `"NNPA"`, `"CPU"`, or empty string
- **`quantize`** (optional): Enable quantization for the operation
  - Values: `true`, `false`

### Pattern Matching

Patterns are processed in order, and the first matching pattern wins. Use regex for flexible matching:

```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Conv",
          "onnx_node_name": "conv_layer_1"
        },
        "rewrite": {
          "device": "CPU"
        }
      }
    },
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
    }
  ]
}
```

In this example, `conv_layer_1` runs on CPU, while all other Conv operations run on NNPA with quantization.

## Best Practices

1. **Version Control**: Store config files in version control for reproducible builds
2. **Environment-Specific Configs**: Use different config files for development, testing, and production
3. **Minimal Configs**: Only specify options that differ from defaults
4. **Comments**: While JSON doesn't support comments, use descriptive file names (e.g., `config-production.json`)
5. **Validation**: Test your config file with `--help` to verify options are parsed correctly:
   ```bash
   onnx-mlir --config-file=config.json --help
   ```

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
To see which options were loaded from the config:
```bash
onnx-mlir --config-file=config.json model.onnx
# Output: Loaded N compile option(s) from config.json
```

## Related Documentation

- [NNPA Accelerator Configuration](JsonConfigFile-NNPA.md)
- [Compiler Options](Options.md)
- [Building onnx-mlir](BuildOnLinuxOSX.md)

## Migration from Command Line

If you have a complex command line:
```bash
onnx-mlir -O3 -march=z16 -mcpu=z16 --enable-parallel -j 8 model.onnx
```

Convert it to a config file:
```json
{
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16", "--enable-parallel", "-j", "8"]
}
```

Then use:
```bash
onnx-mlir --config-file=config.json model.onnx
```
