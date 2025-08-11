# Exit immediately if a command exits with a non-zero status.
set -e

export INSTALL_PROTOBUF_PATH=~/work/protobuf_install # Changed to a dedicated install directory
export BUILD_TYPE=Release
export CORE_NUMBER=1

# Build protobuf from source with -fPIC on Unix-like system
ORIGINAL_PATH=$(pwd)
cd ..
wget https://github.com/abseil/abseil-cpp/releases/download/20230802.2/abseil-cpp-20230802.2.tar.gz
tar -xvf abseil-cpp-20230802.2.tar.gz

wget https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz
tar -xvf protobuf-25.1.tar.gz
cd protobuf-25.1
mkdir build_source && cd build_source
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PROTOBUF_PATH -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DABSL_ROOT_DIR="${ORIGINAL_PATH}/../abseil-cpp-20230802.2" -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=on ..
if [ "$INSTALL_PROTOBUF_PATH" == "/usr" ]; then
    # Don't use sudo for root
    if [[ "$(id -u)" == "0" ]]; then
      cmake --build . --target install --parallel $CORE_NUMBER
    else
      # install Protobuf on default system path so it needs sudo permission
      sudo cmake --build . --target install --parallel $CORE_NUMBER
    fi
else
    cmake --build . --target install --parallel $CORE_NUMBER
    export PATH=$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$INSTALL_PROTOBUF_PATH/bin:$PATH
    export LDFLAGS="-L$INSTALL_PROTOBUF_PATH/lib"
    export CPPFLAGS="-I$INSTALL_PROTOBUF_PATH/include"
fi
protoc --version
cd $ORIGINAL_PATH