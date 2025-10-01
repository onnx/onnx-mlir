# This script is to run GPT2 from HuggingFace.
# Command: ONNX_MLIR_HOME=/workdir/onnx-mlir/build/Debug python gpt2-decode.py -m /workdir/onnx-mlir/utils/python/transformers -o 100 2>&1 | tee log.txt
#
# When running this script for the first time, it will download onnx models from
# HuggingFace and compile the models. The onnx models and compiled models are
# cached in the current folder (by default).
#
# Change compile_flags if targeting a different machine.
################################################################################

import os
import sys
import time
import json
import requests as req
from urllib.request import urlretrieve
import argparse

import numpy as np
from transformers import AutoTokenizer

from omdecoder import OMModelForCausalLM

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
sys.path.append(RUNTIME_DIR)
try:
    from PyCompileAndRuntime import OMCompileExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntimeC target, build it by running `make PyRuntimeC`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntimeC` outputs to `build/Debug` by default"
    )

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="", help="Prompt")
parser.add_argument(
    "-o", "--out_tokens", type=int, default=100, help="Number of output tokens"
)
parser.add_argument(
    "-i", "--iterations", type=int, default=1, help="Number of iterations"
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="/workdir/onnx-mlir/utils/python/transformers",
    help="Path to compiled model directory",
)
args = parser.parse_args()

# Information to download onnx models from HuggingFace.
model_name_or_path = "gpt2"  # can be gpt2, gpt2-medium, gpt2-large
decoder_model_name = "decoder_model.onnx"
decoder_with_past_model_name = "decoder_with_past_model.onnx"
config_json_name = "config.json"
decoder_url = f"https://huggingface.co/openai-community/{model_name_or_path}/resolve/main/onnx/{decoder_model_name}"
decoder_data_url = f"https://huggingface.co/openai-community/{model_name_or_path}/resolve/main/onnx/{decoder_model_name}_data"
decoder_with_past_url = f"https://huggingface.co/openai-community/{model_name_or_path}/resolve/main/onnx/{decoder_with_past_model_name}"
decoder_with_past_data_url = f"https://huggingface.co/openai-community/{model_name_or_path}/resolve/main/onnx/{decoder_with_past_model_name}_data"
config_json_url = f"https://huggingface.co/openai-community/{model_name_or_path}/resolve/main/onnx/{config_json_name}"

# Local directories for caching the model.
cache_dir = "./"
decoder_model_path = f"{cache_dir}/{decoder_model_name}"
decoder_data_path = f"{cache_dir}/{decoder_model_name}_data"
decoder_with_past_model_path = f"{cache_dir}/{decoder_with_past_model_name}"
decoder_with_past_data_path = f"{cache_dir}/{decoder_with_past_model_name}_data"
config_json_path = f"{cache_dir}/{config_json_name}"

# Download the model to a local dir.
if not os.path.exists(decoder_model_path):
    print(f"Downloading {decoder_url}")
    urlretrieve(decoder_url, decoder_model_path)
    print("Done")
if not os.path.exists(decoder_data_path):
    if req.head(decoder_data_url, allow_redirects=True).status_code == 200:
        print(f"Downloading {decoder_data_url}")
        urlretrieve(decoder_data_url, decoder_data_path)
        print("Done")
if not os.path.exists(decoder_with_past_model_path):
    print(f"Downloading {decoder_with_past_url}")
    urlretrieve(decoder_with_past_url, decoder_with_past_model_path)
    print("Done")
if not os.path.exists(decoder_with_past_data_path):
    if req.head(decoder_with_past_data_url, allow_redirects=True).status_code == 200:
        print(f"Downloading {decoder_with_past_data_url}")
        urlretrieve(decoder_with_past_data_url, decoder_with_past_data_path)
        print("Done")
if not os.path.exists(config_json_path):
    print(f"Downloading the config json file {config_json_url}")
    urlretrieve(config_json_url, config_json_path)
    print("Done")

with open(config_json_path) as f:
    cfg = json.load(f)
    print("Model configuration: {}\n".format(cfg))
    num_attention_heads = cfg["n_head"]
    hidden_size = cfg["n_embd"]
    num_layers = cfg["n_layer"]
    eos_token_id = cfg["eos_token_id"]

# Create CompileExecutionSession to compile the models
compile_flags = "-O3 -v --onnx-op-stats TXT"
# compile_flags = "-O3 --march=z16 --maccel=NNPA -v --onnx-op-stats TXT"
decoder_sess = OMCompileExecutionSession(
    decoder_model_path, compile_flags + " -tag=decoder", reuse_compiled_model=1
)
decoder_with_past_sess = OMCompileExecutionSession(
    decoder_with_past_model_path,
    compile_flags + " -tag=decoder_with_past",
    reuse_compiled_model=1,
)

# Setup a tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, cache_dir=cache_dir, local_files_only=True
)

# Load model (after compilation is done)
model_dir = args.model
model = OMModelForCausalLM.from_pretrained(model_id=model_dir)

if args.prompt == "":
    prompt = "The future of artificial intelligence is"
else:
    prompt = args.prompt

max_length = args.out_tokens

# Tokenize input & convert to numpy arrays
model_inputs_dict = tokenizer(prompt, return_tensors="np")
model_inputs = {
    "input_ids": model_inputs_dict["input_ids"].astype(np.int64),
    "attention_mask": model_inputs_dict["attention_mask"].astype(np.int64),
}

original_inputs_len = model_inputs["input_ids"].shape[1]

print("Prompt:", prompt)

t = []
for i in range(args.iterations):
    t0 = time.time()

    # Generate with numpy arrays
    output = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=max_length,
        do_sample=True,
        top_k=10,
        temperature=1.2,
    )

    t_elap = time.time() - t0
    t += [t_elap]

    # Decode the full generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(generated_text)

    # Also show just the new tokens
    new_tokens = output[0, original_inputs_len:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("\nNew tokens only:")
    print(new_text)

    # Extract only the new tokens for timing calculations
    output_ids = new_tokens

real_output_len = len(output_ids)
print("\nQuery Info:")
print(" Input tokens:", original_inputs_len)
print(" Output tokens:", real_output_len)
print(" Times", t)
t_elap = np.min(t)
print(" t_elap: %.2f seconds" % (t_elap))
print(
    " Latency: {} msec/token, throughput: {} tokens/sec".format(
        t_elap / real_output_len * 1000, real_output_len / t_elap
    )
)
