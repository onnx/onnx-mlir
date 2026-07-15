# SPDX-License-Identifier: Apache-2.0

##################### test_decoder_models.py ###################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch_onnxmlir
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH

logger = logging.basicConfig(level=logging.INFO)

model_name = os.environ["DECODER_MODEL"]
if model_name == "gpt2":
    model_path = "openai-community/gpt2"
elif model_name == "Qwen2-0.5B-Instruct":
    model_path = "Qwen/Qwen2-0.5B-Instruct"
elif model_name == "granite-3.3-2b-instruct":
    model_path = "ibm-granite/granite-3.3-2b-instruct"
elif model_name == "granite-4.0-350M":
    model_path = "ibm-granite/granite-4.0-350M"
else:
    assert False, "Wrong arguments"


class TestDecoderModel(TorchOMTestCase):

    def test_decoder_model(self):
        # Load model and tokenizer.
        # attn_implementation="eager" is needed to decompose sdpa ops.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            dtype="float32",
            cache_dir=self.TMP_DIR,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.TMP_DIR)

        # Turn the model to inference mode.
        model.eval()

        # Compile the model for NNPA.
        om_options = {
            "compiler_image_name": COMPILER_IMAGE_NAME,
            "compiler_path": COMPILER_PATH,
            "compile_options": "-O3 -march=z16 -maccel=NNPA",
        }
        model.forward = torch.compile(
            model.forward,
            backend="onnxmlir",
            options=om_options,
        )

        # Prepare inputs.
        if model_name == "gpt2":
            prompt = "Hello, I'm a language model,"
        else:
            prompt = "The capital of Japan is"
        input_tokens = tokenizer(prompt, return_tensors="pt")
        original_inputs_len = len(input_tokens["input_ids"][0])

        # Generate output tokens.
        with self.assertLogs(logger) as cm:
            with torch.no_grad():
                output = model.generate(**input_tokens, max_new_tokens=11)
        # Verify that there is no eager mode.
        self.assertNoEagerMode(cm.output)
        # Verify that the model is compiled twice for prefill and decode phases.
        self.assertNumCompile(cm.output, 2)

        output_ids = output[0, original_inputs_len:]
        answer = tokenizer.decode(output_ids)
        print(answer)

        # Verify the output sentence.
        if model_name == "gpt2":
            ref_answer = " and I'm not a programmer. I'm a programmer"
            assert answer == ref_answer, "Wrong answer"
        elif model_name == "granite-4.0-350M":
            ref_answer = " Tokyo, which is located on the eastern coast of the"
            assert answer == ref_answer, "Wrong answer"


if __name__ == "__main__":
    testcase = TestDecoderModel()
    testcase.test_decoder_model()
