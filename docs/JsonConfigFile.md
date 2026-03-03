<!--- SPDX-License-Identifier: Apache-2.0 -->

# Overview

The `onnx-mlir` compiler supports loading configuration from a JSON file, allowing you to specify compile options and operation-specific settings in a single, reusable configuration file.

# Features

<<<<<<< HEAD
- **Compile Options**: Specify command-line options as an array in the config file.
- **Default Config**: Automatically loads `omconfig.json` from the same directory as the input model if present.
- **Custom Config**: Use `--config-file` to specify a custom configuration file.
- **Config Override**: Config file options override command-line settings.
=======
- **Compile Options**: Specify command-line options as an array in the config file. These are **appended** to the options provided on the command line.
- **Default Config**: Automatically loads `omconfig.json` from the same directory as the input model if present.
- **Custom Config**: Use `--config-file` to specify a custom configuration file.
>>>>>>> main

# JSON Scheme Description

**Example:**
```json
{
  "compile_options": ["-O3", "-march=z16", "-mcpu=z16"]
}
```

## Top-Level Keys

| Key              | Type                       | Description                                                                                       |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| compile_options  | array of string (optional) | A list of compiler command-line options that are appended to the existing command-line arguments. |

**Example:**
```json
"compile_options": ["-O3", "-march=z16", "--enable-parallel"]
```

**Supported Options:**
- Optimization levels: `-O0`, `-O1`, `-O2`, `-O3`
- Target architecture: `-march=<arch>`, `-mcpu=<cpu>`, `-mtriple=<triple>`
- Accelerator: `-maccel=NNPA`
- Parallelization: `--enable-parallel`
- And all other onnx-mlir command-line options

# Usage

## Using Default Config File

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

## Using Custom Config File

```bash
onnx-mlir --config-file=my-config.json model.onnx
```

## Config File Overrides CLI Options

Config file options take precedence over command-line settings:

```bash
# CLI has -O2, but config file overrides to -O3
onnx-mlir -O2 --config-file=config.json model.onnx
# Result: Uses -O3 from config file
```

# Migration from Command Line

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

# Troubleshooting

## Config File Not Found
If the config file doesn't exist, onnx-mlir continues without error (unless you explicitly specify `--config-file`).

## Parse Errors
If the JSON is malformed, you'll see an error message:
```
Warning: Failed to parse config file: config.json
```

## Invalid Options
If the config contains invalid compiler options, you'll see the standard option parsing error.

## Debugging
To see which options were loaded from the config, use `-v`:
```bash
onnx-mlir --config-file=config.json model.onnx -v
# Output contains the following information:
# Config file: config.json
# Onnx-mlir command: ./Debug/bin/onnx-mlir --config-file=config.json model.onnx -v -O2 -march=z16
```

# Related Documentation
- [Compiler Options](Options.md)
