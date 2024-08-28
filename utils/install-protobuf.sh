# Check out protobuf
PROTOBUF_VERSION=21.12
git clone -b v${PROTOBUF_VERSION} --depth 1 --recursive https://github.com/protocolbuffers/protobuf.git

cd protobuf
./autogen.sh
./configure --enable-static=no  --prefix=/usr
sudo make -j2 install

# Doesn't work on Ubuntu, just needed for MacOS?
#cd python
#python3 setup.py install --cpp_implementation
