# Check out protobuf (mac x86 seems to need an older version)
PROTOBUF_VERSION=3.18.3
git clone -b v${PROTOBUF_VERSION} --depth 1 --recursive https://github.com/protocolbuffers/protobuf.git

cd protobuf
./autogen.sh
./configure --enable-static=no
make -j$(sysctl -n hw.logicalcpu) install
cd python
python3 setup.py install --cpp_implementation
