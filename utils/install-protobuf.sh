ABSL_VERSION=20250127.0
ABSL_URL=https://github.com/abseil/abseil-cpp.git
ABSL_DIR=abseil-cpp
git clone -b ${ABSL_VERSION} --depth 1 --recursive ${ABSL_URL} ${ABSL_DIR}
mkdir -p ${ABSL_DIR}/build
cd ${ABSL_DIR}/build
cmake -DCMAKE_INSTALL_PREFIX=/usr \
       -DABSL_PROPAGATE_CXX_STD=ON \
       -DABSL_BUILD_TESTING=OFF \
       -DBUILD_SHARED_LIBS=ON ..
sudo make -j2 install
cd ../..

PROTOBUF_VERSION=4.25.3
PROTOBUF_URL=https://github.com/protocolbuffers/protobuf.git
PROTOBUF_DIR=protobuf
git clone -b v${PROTOBUF_VERSION} --depth 1 --recursive ${PROTOBUF_URL} ${PROTOBUF_DIR}
mkdir -p ${PROTOBUF_DIR}/build
cd ${PROTOBUF_DIR}/build
# Must specify -Dprotobuf_BUILD_TESTS=OFF otherwise find_package(absl)
# in onnx will fail due to missing protobuf::gmock target
cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DBUILD_SHARED_LIBS=ON \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_ABSL_PROVIDER=package ..
sudo make -j2 install

# Doesn't work on Ubuntu, just needed for MacOS?
#cd python
#python3 setup.py install --cpp_implementation
