# Install rust, cargo, and cargo-bazel
RUST_VERSION=1.89
CARGO_BAZEL_VERSION=0.16.0
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
     sh -s -- -y --default-toolchain none
. "${HOME}/.cargo/env"
rustup install ${RUST_VERSION}
cargo install cargo-bazel --version ${CARGO_BAZEL_VERSION}

# Install absl for new version of protobuf
ABSL_VERSION=20240722.1
ABSL_URL=https://github.com/abseil/abseil-cpp.git
ABSL_DIR=abseil-cpp
git clone -b ${ABSL_VERSION} --recursive ${ABSL_URL} ${ABSL_DIR}
mkdir -p ${ABSL_DIR}/build
cd ${ABSL_DIR}/build
cmake -DCMAKE_INSTALL_LIBDIR=lib \
      -DABSL_PROPAGATE_CXX_STD=ON \
      -DBUILD_SHARED_LIBS=ON ..
make -j${NPROC} install
cd ../..
rm -rf ${ABSL_DIR} ${HOME}/.cache

# Install protobuf
PROTOBUF_VERSION=6.31.1
PROTOBUF_URL=https://github.com/protocolbuffers/protobuf.git
PROTOBUF_DIR=protobuf
git clone -b v${PROTOBUF_VERSION} --recursive ${PROTOBUF_URL} ${PROTOBUF_DIR}
mkdir -p ${PROTOBUF_DIR}/build
cd ${PROTOBUF_DIR}/build
# Must specify -Dprotobuf_BUILD_TESTS=OFF otherwise find_package(absl)
# in onnx will fail due to missing protobuf::gmock target
cmake -DCMAKE_INSTALL_LIBDIR=lib \
      -DBUILD_SHARED_LIBS=ON \
      -Dprotobuf_BUILD_TESTS=OFF ..
make -j${NPROC} install
cd ..
# New version of python protobuf can no longer be built with setup.py.
# Must use bazel to build. protobuf v6.31.1 is the first version using
# rules_rust 0.56.0 which has s390x support. rules_buf still needs a
# small patch.
CARGO_BAZEL_GENERATOR_URL=file:///root/.cargo/bin/cargo-bazel
CARGO_BAZEL_REPIN=true
bazel build //python/dist:binary_wheel
pip3 install bazel-bin/python/dist/protobuf-${PROTOBUF_VERSION}-*.whl
cd ..
rm -rf ${PROTOBUF_DIR} ${HOME}/.cache
