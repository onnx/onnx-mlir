# Command to run: OMP_NUM_THREADS=6 python test_bert.py
import torch
import logging
import time

from transformers import AutoModel, AutoTokenizer
import torch_onnxmlir

logging.basicConfig(level=logging.INFO)  # Or INFO, WARNING, etc.

# model_path = "google-bert/bert-base-uncased"
model_path = "ibm-granite/granite-embedding-30m-english"
# model_path = "ibm-granite/granite-embedding-278m-multilingual"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Turn to inference mode.
model.eval()

# Compile the model
om_options = {
    "compiler_image_name": None,
    "compile_options": "-O3 -march=z17 -maccel=NNPA --parallel --enable-zhigh-decompose-stick-unstick",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}

compiled_model = torch.compile(
    model,
    backend="onnxmlir",
    options=om_options,
)


def get_cls_embedding(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token


def get_cls_embedding_onnxmlir(inputs):
    with torch.no_grad():
        outputs = compiled_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token


text1 = "I love machine learning."
text1_tok = tokenizer(text1, return_tensors="pt")
text2 = "AI is fascinating."
text2_tok = tokenizer(text2, return_tensors="pt", max_length=256, padding="max_length")
print("text1: ", text1)
print("text2: (padding to 256)", text2)

start = time.time()
emb1 = get_cls_embedding(text1_tok)
print(f"[pytorch-cpu], embedding text1 took", (time.time() - start) * 1000, "ms")

start = time.time()
emb2 = get_cls_embedding(text2_tok)
print(f"[pytorch-cpu], embedding text2 took", (time.time() - start) * 1000, "ms")
similarity = torch.nn.functional.cosine_similarity(emb1, emb2)

start = time.time()
emb1_onnxmlir = get_cls_embedding_onnxmlir(text1_tok)
print(f"[onnxmlir-zaiu], embedding text1 took", (time.time() - start) * 1000, "ms")

start = time.time()
emb2_onnxmlir = get_cls_embedding_onnxmlir(text2_tok)
print(f"[onnxmlir-zaiu], embedding text2 took", (time.time() - start) * 1000, "ms")

similarity_onnxmlir = torch.nn.functional.cosine_similarity(
    emb1_onnxmlir, emb2_onnxmlir
)

similarity_torch_onnxmlir1 = torch.nn.functional.cosine_similarity(emb1, emb1_onnxmlir)
similarity_torch_onnxmlir2 = torch.nn.functional.cosine_similarity(emb2, emb2_onnxmlir)
print("Cosine similarity [pytorch] text1 vs text2:", similarity.item())
print("Cosine similarity [onnxmlir] text1 vs text2:", similarity_onnxmlir.item())
print(
    "Cosine similarity [text1] pytorch vs onnxmlir:", similarity_torch_onnxmlir1.item()
)
print(
    "Cosine similarity [text2] pytorch vs onnxmlir:", similarity_torch_onnxmlir2.item()
)
