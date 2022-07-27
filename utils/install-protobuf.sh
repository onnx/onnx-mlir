# Check out a specific tag v3.16.0 which is the recommended version of onnx 1.11.0
PROTOBUF_VERSION=3.16.0
git clone -b v${PROTOBUF_VERSION} --recursive https://github.com/google/protobuf.git

cd protobuf
./autogen.sh
./configure --enable-static=no
make -j$(sysctl -n hw.logicalcpu) install
cd python
python3 setup.py install
