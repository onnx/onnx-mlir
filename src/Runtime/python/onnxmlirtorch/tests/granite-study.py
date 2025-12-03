import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy

device = "cpu"
model_path = "ibm-granite/granite-3.1-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# Added code!
import onnxmlirtorch

onnxmlirtorch.interceptForward(model)

# change input text as desired
chat = [
    {
        "role": "user",
        "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location.",
    },
]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)

output = model.generate(**input_tokens, max_new_tokens=100)

# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
