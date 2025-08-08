# Check out protobuf source code and build and install it
set -e 
PROTOBUF_VERSION=25.1
INSTALL_PROTOBUF_PATH=~/work
git clone -b v${PROTOBUF_VERSION} --depth 1 --recursive https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
mkdir build_source && cd build_source
cmake -G Ninja ../ \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PROTOBUF_PATH \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DABSL_PROPAGATE_CXX_STD=ON \
    ..
cmake --build . --target install
cd ~/work/protobuf/python && python3 setup.py install --cpp_implementation
export PATH=$INSTALL_PROTOBUF_PATH/protobuf/include:$INSTALL_PROTOBUF_PATH/protobuf/lib:$INSTALL_PROTOBUF_PATH/protobuf/bin:$PATH
protoc --version
echo "protobuf installed"
