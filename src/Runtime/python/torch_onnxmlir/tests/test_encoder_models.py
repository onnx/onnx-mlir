# SPDX-License-Identifier: Apache-2.0

##################### test_encoder_models.py ###################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import os
import torch
import logging
from transformers import AutoModel, AutoTokenizer

import torch_onnxmlir
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH

logger = logging.basicConfig(level=logging.INFO)

model_name = os.environ["ENCODER_MODEL"]

if model_name == "granite-embedding-30m-english":
    model_path = "ibm-granite/granite-embedding-30m-english"
elif model_name == "granite-embedding-278m-multilingual":
    model_path = "ibm-granite/granite-embedding-278m-multilingual"
elif model_name == "bert-base-uncased":
    model_path = "google-bert/bert-base-uncased"
else:
    assert False, "Wrong arguments"


def get_cls_embedding(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token


class TestEncoderModel(TorchOMTestCase):

    def test_encoder_model(self):
        torch_onnxmlir.config.cache_dir = self.TMP_DIR

        # Load model and tokenizer.
        model = AutoModel.from_pretrained(
            model_path, dtype="float32", cache_dir=self.TMP_DIR
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
        compiled_model = torch.compile(
            model,
            backend="onnxmlir",
            options=om_options,
        )

        # Prepare inputs.
        text1 = "I love machine learning."
        text1_tok = tokenizer(text1, return_tensors="pt")
        text2 = "AI is fascinating."
        text2_tok = tokenizer(
            text2, return_tensors="pt", max_length=256, padding="max_length"
        )

        # Reference output from pytorch using CPU.
        emb1 = get_cls_embedding(model, text1_tok)
        emb2 = get_cls_embedding(model, text2_tok)

        # Use torch.compile for NNPA.
        with self.assertLogs(logger) as cm:
            emb1_onnxmlir = get_cls_embedding(compiled_model, text1_tok)
        self.assertCompile(cm.output)

        with self.assertLogs(logger) as cm:
            emb2_onnxmlir = get_cls_embedding(compiled_model, text2_tok)
        self.assertInCache(cm.output)

        similarity_torch_onnxmlir1 = torch.nn.functional.cosine_similarity(
            emb1, emb1_onnxmlir
        )
        similarity_torch_onnxmlir2 = torch.nn.functional.cosine_similarity(
            emb2, emb2_onnxmlir
        )
        assert similarity_torch_onnxmlir1.item() > 0.98, "Accuracy does not match"
        assert similarity_torch_onnxmlir2.item() > 0.98, "Accuracy does not match"
