# Check out protobuf v3.20.2
PROTOBUF_VERSION=3.20.2
git clone -b v${PROTOBUF_VERSION} --recursive https://github.com/google/protobuf.git

cd protobuf
./autogen.sh
./configure --enable-static=no
make -j$(sysctl -n hw.logicalcpu) install
cd python
python3 setup.py install --cpp_implementation
