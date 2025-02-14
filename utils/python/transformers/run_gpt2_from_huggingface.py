#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### run_gpt2_from_huggingface.py #############################
#
# Copyright 2019-2024 The IBM Research Authors.
#
################################################################################
#
# This script is to run GPT2 from HuggingFace.
# Command: ONNX_MLIR_HOME=/workdir/onnx-mlir/build/Debug python run_gpt2_from_huggingface.py 2>&1 | tee log.txt
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

import numpy as np
from transformers import AutoTokenizer

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
sys.path.append(RUNTIME_DIR)
try:
    from PyCompileAndRuntime import OMCompileExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
    )

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

# Create CompileExecutionSession to compile and run the model,
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
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

# Tokenize the input text.
prompt_text = "Which is the highest mountain in Japan?"
pt_inputs = tokenizer(prompt_text, return_tensors="pt")
output_length = 13

# Generate tokens.
ts = []
num_runs = 3
for r in range(num_runs):
    output_ids = []
    kv_cache = None

    t = 0
    attention_mask = pt_inputs["attention_mask"].numpy()
    inputs = [pt_inputs["input_ids"].numpy(), attention_mask]
    for step in range(output_length):
        t0 = time.time()
        if kv_cache is None:
            outputs = decoder_sess.run(inputs)
        else:
            outputs = decoder_with_past_sess.run(inputs)
        t_elap = time.time() - t0
        t += t_elap
        # Greedy approach is used here.
        logits = outputs[0][:, -1, :]
        next_id = np.argmax(logits, axis=1, keepdims=True)
        kv_cache = outputs[1:]
        # Only for batchsize = 1
        attention_mask = np.append(
            attention_mask, np.array([[1]], dtype=np.int64), axis=1
        )
        inputs = [next_id] + kv_cache + [attention_mask]
        output_ids += [next_id[0][0]]

    ts += [t]
    if r == num_runs - 1:
        # Expected answer: "The highest mountain in Japan is the Mt. Fuji."
        print("Prompt: {}".format(prompt_text))
        print("Generated words: {}\n".format(tokenizer.decode(output_ids).strip()))

print("times", ts)
t = np.min(ts)
print("t_elap: %.2f seconds" % (t))
print(
    "Latency: {} msec/token, thoughput: {} tokens/sec".format(
        t / output_length * 1000, output_length / t
    )
)
