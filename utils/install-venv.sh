#!/usr/bin/bash
set -e

python3 -m pip install --upgrade wheel
python3 -m pip install -r requirements.txt
pip install onnx==1.15.0
