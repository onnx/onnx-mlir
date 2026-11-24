<!--- SPDX-License-Identifier: Apache-2.0 -->

# Overview
The script, export_attention_onnx.py, is intended to export transformer models in Hugging Face to onnx model with AttentionOp. The exporting could use the pre-trained weights, or just random weight for development. Both way will keep the basic
feature of each layer, such as attention heads, and number of states.

-- Use pre-trained weight. Use the name card of the Hugging Face model to the `--model` parameter. The script will download the model from Hugging Face and export to onnx. For example, granite 4.1 decoder with static cache with dynamic size: `python export_attention_onnx.py --model ibm-granite/granite-4.1-3b --static-cache --dynamic-cache-len --no-optimize`
-- Use random weigth. The script support several models, such as "bert", "gpt2", "llama", "granite", and "modernbert". The script will use the configure class of these model to create onnx model. The weights are random values. If option "--compact" is used, each weight will have one scalar value and the size of the onnx model will be very small.

The script also provides the driver to run the model. It compares the execution output from PyTorch and onnxruntime to verify the correctness. So far the execution driver is NOT used independently because it needs to know how the model is exported, not just the onnx file. 

# Control exporting
Use "python export_attention_onnx.py -h" to list all the options.  

## Examples
I use gpt2 as the example because it is MHA, while the granite is GQA.
* gpt2 without kv cache: `python export_attention_onnx.py --random gpt2 --compact --no-optimize`. The result is gpt2_opset23.onnx. There is no kv cache in the input or output in the exported model.
* gpt2 with kv_cache: `python export_attention_onnx.py --random gpt2 --compact --no-optimize --kv-cache`. The result is gpt2_kv_opset23.onnx. The input and output parameters include kv cache but TensorScatter is not used for kv cache update.
* gpt2 with static kv_cache: `export_attention_onnx.py --random gpt2 --compact --no-optimize --kv-cache --static-cache --dynamic-cache-len`. The result is gpt2_static_kv_opset24.onnx. The TensorScatter op is used for kv cache update and the size of static cache is dynamic. In this model, position index is used for Attention op.
* gpt2 with static kv_cache to use mask for attention op: `export_attention_onnx.py --random gpt2 --compact --no-optimize --kv-cache --static-cache --dynamic-cache-len --attn-mask`. The default  gpt2_static_kv_mask_opset24.onnx.  The TensorScatter op is used for kv cache update and the size of static cache is dynamic. In this model, mask is used for Attention op.
* gpt2 with sliding window attention and circular cache: `python export_attention_onnx.py --random gpt2 --compact --static-cache --sliding-window 128 --max-cache-len 128 --circular-cache --seq 0`. This model is not verified by onnxruntime. The IR from onnx-mlir looks fine to me.
