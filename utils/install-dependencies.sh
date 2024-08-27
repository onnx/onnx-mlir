#!/usr/bin/bash
set -e

# Keep in sync with utils/install-protobuf.sh
PROTOBUF_VERSION=21.12
git clone -b v${PROTOBUF_VERSION} --depth 1 --recursive https://github.com/protocolbuffers/protobuf.git
cd protobuf
./autogen.sh
./configure --enable-static=no --prefix=/usr
sudo make -j2 install
cd ..

python3 -m pip install --upgrade wheel
python3 -m pip install -r requirements.txt
pip install onnx==1.15.0
