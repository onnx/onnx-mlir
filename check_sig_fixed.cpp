#include <OnnxMlirRuntime.h>
#include <iostream>
#include <dlfcn.h>

int main() {
    void* handle = dlopen("./test_linalg_matmul.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot load library: " << dlerror() << std::endl;
        return 1;
    }

    typedef const char* (*SigFunc)(const char*);
    SigFunc inputSig = (SigFunc)dlsym(handle, "omInputSignature");
    SigFunc outputSig = (SigFunc)dlsym(handle, "omOutputSignature");
    
    if (inputSig) {
        const char* sig = inputSig("run_main_graph");
        std::cout << "Input signature: " << (sig ? sig : "null") << std::endl;
    } else {
        std::cerr << "omInputSignature not found" << std::endl;
    }
    if (outputSig) {
        const char* sig = outputSig("run_main_graph");
        std::cout << "Output signature: " << (sig ? sig : "null") << std::endl;
    } else {
        std::cerr << "omOutputSignature not found" << std::endl;
    }
    
    dlclose(handle);
    return 0;
}
