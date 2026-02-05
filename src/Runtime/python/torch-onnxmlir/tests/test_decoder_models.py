# Usage: MODEL=granite4 && OMP_NUM_THREADS=6 TORCHONNXMLIR_CACHE_DIR=./cache_${MODEL} python ./generate.py ${MODEL} 2>&1 | tee ${MODEL}.log

import sys
import time
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

import torch_onnxmlir
import logging

# logging.basicConfig(level=logging.INFO)  # Or INFO, WARNING, etc.

torch_onnxmlir.config.session_cache_limit = 200
torch_onnxmlir.config.same_hash_size = 0

model_name = sys.argv[1]
if model_name == "gpt2":
    model_path = "openai-community/gpt2"
elif model_name == "qwen":
    model_path = "Qwen/Qwen2-0.5B-Instruct"
elif model_name == "granite3":
    model_path = "ibm-granite/granite-3.3-2b-instruct"
elif model_name == "granite4":
    model_path = "ibm-granite/granite-4.0-350M"
else:
    print("Usage: generate.py gpt2/qwen/granite3/granite4")
    sys.exit()

tokenizer = AutoTokenizer.from_pretrained(model_path)
# attn_implementation="eager" is needed to decompose sdpa ops
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
model.eval()

om_options = {
    "compiler_image_name": None,
    "compile_options": "-O3 -march=z17 -maccel=NNPA --parallel --printONNXBasicIR=10 -v --onnx-op-stats TXT",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}
model.forward = torch.compile(
    model.forward,
    backend="onnxmlir",
    options=om_options,
)

# Change input text as desired
content = "Please list one IBM Research laboratory located in the United States. You should only output its name and location."
chat = [
    {"role": "user", "content": content},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# prompt = "What is quantum computing?"

print(f"Prompt: {content}")
input_tokens = tokenizer(prompt, return_tensors="pt")
original_inputs_len = len(input_tokens["input_ids"][0])

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generate output tokens
t = []
for i in range(3):
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**input_tokens, max_new_tokens=100, streamer=streamer)
    end = time.perf_counter()
    t_elap = end - start
    t += [t_elap]
    output_ids = output[0, original_inputs_len:]

real_output_len = len(output_ids)
print("\nQuery Info:")
print(" Input tokens:", original_inputs_len)
print(" Output tokens:", real_output_len)
print(" Times", t)

t_elap = np.min(t)
print(" Best elapsed time: %.2f seconds" % (t_elap))
print(
    " Latency: {} msec/token, throughput: {} tokens/sec".format(
        t_elap / real_output_len * 1000, real_output_len / t_elap
    )
)
