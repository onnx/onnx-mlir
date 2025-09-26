# Inferencing downloaded ONNX Models using Python

This folder provides Python scripts for to downloading, compiling and inferencing autoregressive transformer models like GPT-2, Granite 3, or other models of the same architecture, using ONNX-MLIR.

### Prerequisites

It is good practice to first setup a virtual environment inside the `onnx-mlir` directory to isolate dependencies and avoid potential version conflicts with other projects.
```bash
python -m venv venv
source venv/bin/activate
```

Then, if you haven't done so already, you can install all the required dependencies:

```bash
pip install -r requirements.txt
```

Also, just like in the [mnist-example](../../../docs/mnist_example/README.md), it is recommended to update your environment variables like this:

```bash
# ONNX_MLIR_ROOT points to the root of the onnx-mlir,
# under which the include and the build directory lies.
export ONNX_MLIR_ROOT=$(pwd)/../..
# Define the bin directory where onnx-mlir binary resides.Change only if you
#have a non - standard install.
export ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin
# Define the include directory where onnx-mlir runtime include files resides.
#Change only if you have a non - standard install.
export ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include

# Include ONNX-MLIR executable directories part of $PATH.
export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH

# Compiler needs to know where to find its runtime. 
# Set ONNX_MLIR_RUNTIME_DIR to proper path.
export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
```

Alternatively, you can use a .env file. If you are using the default container setup, just copy the cell below, create a `.env`-file inside this folder, and paste the content.

```bash
ONNX_MLIR_ROOT=/workdir/onnx-mlir
ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin
ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include
ONNX_MLIR_RUNTIME_DIR=$ONNX_MLIR_ROOT/build/Debug/lib
ONNX_MLIR_HOME=$ONNX_MLIR_ROOT/build/Debug
```

Once done, you can set all the environment variables with one command:
```bash
set -a && source .env && set +a
```

Lastly, set the `PYTHONPATH` variable so Python knows from where to import the necessary runtime module. You can do this by setting the PYTHONPATH to the following path:
```bash
export PYTHONPATH=/workdir/onnx-mlir/build/Debug/lib:$PYTHONPATH
```

### Using Custom Decoding Algorithms

For text generation with Large Language Models, there is a wide variety of decoding algorithms that all work a little differently. HuggingFace `transformers` offers a lot of algorithms and a lot of possible configurations. However, for ONNX-MLIR, the overall goal is to have the decoding algorithms as part of the compiler, which is not possible using transformers.

That is why the module `omgenerate` offers an adapted and simplified version of transformers' `GenerationMixin` class, using only Python and NumPy for numerical calculations. Currently supported decoding methods are:
- Greedy Decoding
- Top-k Sampling
- Top-p (nucleus) Sampling
- Temperature Sampling

The `OMGeneration`-class can be used to to build own models, e.g. for ONNX-MLIR inference, by inheriting from the Mixin class:

```python
class OMModelForCausalLM(OMModelDecoder, OMGeneration):
    """ONNX model with a causal language modeling head for ONNX-MLIR inference."""

    main_input_name = "input_ids"
    _supports_cache_class = False
    _is_stateful = True

    def forward(
        self,
        input_ids: np.ndarray = None,
        attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutput:
        # Use instance use_cache if not specified
        use_cache = use_cache if use_cache is not None else self.use_cache
            
        if past_key_values is None or not use_cache:
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=labels,
            )
        else:
            outputs = self.decoder_with_past(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                labels=labels,
            )

        return CausalLMOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )
    
    @classmethod
    def _from_pretrained(cls, model_id: Union[str, Path], config: OMConfig, **kwargs,):
        return super()._from_pretrained(
            model_id, config, init_cls=OMModelForCausalLM, **kwargs
        )
```

Check the full script for the `OMGeneration`-class [here](omgenerate.py), and for the several model classes [here](omdecoder.py).

### Example Script

You can find an usage example in the script [`gpt2-decode.py`](gpt2-decode.py). This script uses the specified class `OMModelForCausalLM` to generate a text sequence with GPT-2. First, it will download the specified ONNX models from huggingface (e.g. gpt2), which will save two `.onnx`-files in your current working directory (`decoder_model` and `decoder_with_past_model`). These 2 ONNX-Graphs will then be automatically compiled to `.so`-files by the ONNX-MLIR compiler. This might take a couple minutes, but you should be able to see some output telling you the compilation status in your command line.

After successful compilation, the script will take the model and generate some tokens based on the defined prompt.

```bash
Command: ONNX_MLIR_HOME=/workdir/onnx-mlir/build/Debug python gpt2-decode.py -m /workdir/onnx-mlir/utils/python/transformers -o 100 2>&1 | tee log.txt 
```

The output should look similar to this:
```shell
Model configuration: {'_name_or_path': 'gpt2', 'activation_function': 'gelu_new', 'architectures': ['GPT2LMHeadModel'], 'attn_pdrop': 0.1, 'bos_token_id': 50256, 'embd_pdrop': 0.1, 'eos_token_id': 50256, 'initializer_range': 0.02, 'layer_norm_epsilon': 1e-05, 'model_type': 'gpt2', 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_inner': None, 'n_layer': 12, 'n_positions': 1024, 'reorder_and_upcast_attn': False, 'resid_pdrop': 0.1, 'scale_attn_by_inverse_layer_idx': False, 'scale_attn_weights': True, 'summary_activation': None, 'summary_first_dropout': 0.1, 'summary_proj_to_labels': True, 'summary_type': 'cls_index', 'summary_use_proj': True, 'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 50}}, 'transformers_version': '4.30.2', 'use_cache': True, 'vocab_size': 50257}

Prompt: The future of artificial intelligence is

Generated text:
The future of artificial intelligence is unclear, and it remains to be seen whether the development and deployment will lead to a more secure AI world.

...

New tokens only:
 unclear, and it remains to be seen whether the development and deployment will lead to a more secure AI world.

...

Query Info:
 Input tokens: 6
 Output tokens: 100
 Times [6.005136966705322]
 t_elap: 6.01 seconds
 Latency: 60.05136966705322 msec/token, throughput: 16.6524095211211 tokens/sec
 ```

If you already have the `.so`-files and run the script again, the model will be cached, so you won't have to go through the whole process again.

You can further modify the inference using the following flags:
| flag | description |
|----------|----------|
| -m    | Path to compiled model directory |
| -o    | Number of output tokens |
| --prompt   | Prompt that is fed to model  |
| -i    | Number of iterations  |

