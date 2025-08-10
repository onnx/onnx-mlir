# Exit immediately if a command exits with a non-zero status.
set -e
# Check out protobuf source code and build and install it
PROTOBUF_VERSION=25.1
INSTALL_PROTOBUF_PATH=~/work/protobuf_install # Changed to a dedicated install directory
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

# Verify that protoc is installed correctly before proceeding
echo "Verifying protoc installation at $INSTALL_PROTOBUF_PATH/bin/protoc..."
if [ -f "$INSTALL_PROTOBUF_PATH/bin/protoc" ]; then
    echo "protoc found."
    "$INSTALL_PROTOBUF_PATH/bin/protoc" --version
else
    echo "Error: protoc not found at $INSTALL_PROTOBUF_PATH/bin/protoc. Installation might have failed."
    exit 1
fi

# Now navigate and run the python setup.py
# Use a subshell to temporarily modify PATH and LDFLAGS for this specific command,
# ensuring our installed protoc and libraries are found first.
(cd ~/work/protobuf/python && \
    PATH="$INSTALL_PROTOBUF_PATH/bin:$PATH" \
    LDFLAGS="-L$INSTALL_PROTOBUF_PATH/lib" \
    CPPFLAGS="-I/$INSTALL_PROTOBUF_PATH/include" \
    python3 setup.py install --cpp_implementation)

# Update the main shell's PATH for subsequent commands like 'protoc --version'
export PATH="$INSTALL_PROTOBUF_PATH/bin:$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$PATH"

protoc --version
echo "protobuf installed"
