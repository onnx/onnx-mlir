/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- runmodel.cpp  ------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
/*
  This file help run a onnx model as simply as possible for testing.
  Compile as follows in the onnx-mlir build subdirectory. The tool is built as
  follows. For dinamically loaded models:

cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh
run-onnx-lib test/backend/test_add.so

  For statically loaded models, best is to run the utility in the directory
  of the model.

cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh test/backend/test_add.so
cd test/backend
run-onnx-lib

  Usage of program is as follows.

Usage: run-onnx-lib [options] model.so

  Program will instantiate the model given by "model.so"
  with random inputs, launch the computation, and ignore
  the results. A model is typically generated by lowering
  an ONNX model using a "onnx-mlir --EmitLib model.onnx"
  command. When the input model is not found as is, the
  path to the local directory is also prepended.

  Options:
    -e name | --entry-point name
         Name of the ONNX model entry point.
         Default is "run_main_graph".
    -n NUM | --iterations NUM
         Number of times to run the tests, default 1
    -v | --verbose
         Print the shape of the inputs and outputs
    -h | --help
         help
*/

//===----------------------------------------------------------------------===//

// Define while compiling.
// #define LOAD_MODEL_STATICALLY 1

#include <algorithm>
#include <assert.h>
#include <dlfcn.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

// Json reader & LLVM suport.
#include "llvm/Support/JSON.h"

// Include ONNX MLIR Runtime support.
#include "OnnxMlirRuntime.h"

using namespace std;

#ifdef _WIN32
// TO BE FIXED
#else
#include <sys/time.h>
#endif
// Data structure to hold measurement times (in microseconds).
vector<uint64_t> timeLogInMicroSec;

// Interface definitions
extern "C" OMTensorList *run_main_graph(OMTensorList *);
extern "C" const char *omInputSignature();
extern "C" const char *omOutputSignature();
extern "C" OMTensor *omTensorCreate(void *, int64_t *, int64_t, OM_DATA_TYPE);
extern "C" OMTensorList *TensorListCreate(OMTensor **, int);
extern "C" void omTensorListDestroy(OMTensorList *list);
// DLL definitions
OMTensorList *(*dll_run_main_graph)(OMTensorList *);
const char *(*dll_omInputSignature)();
const char *(*dll_omOutputSignature)();
OMTensor *(*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
OMTensorList *(*dll_omTensorListCreate)(OMTensor **, int);
void (*dll_omTensorListDestroy)(OMTensorList *);

#if LOAD_MODEL_STATICALLY
#define RUN_MAIN_GRAPH run_main_graph
#define OM_INPUT_SIGNATURE omInputSignature
#define OM_OUTPUT_SIGNATURE omOutputSignature
#define OM_TENSOR_CREATE omTensorCreate
#define OM_TENSOR_LIST_CREATE omTensorListCreate
#define OM_TENSOR_LIST_DESTROY omTensorListDestroy
#define OPTIONS "hn:m:vd:r:"
#else
#define RUN_MAIN_GRAPH dll_run_main_graph
#define OM_INPUT_SIGNATURE dll_omInputSignature
#define OM_OUTPUT_SIGNATURE dll_omOutputSignature
#define OM_TENSOR_CREATE dll_omTensorCreate
#define OM_TENSOR_LIST_CREATE dll_omTensorListCreate
#define OM_TENSOR_LIST_DESTROY dll_omTensorListDestroy
#define OPTIONS "e:hn:m:vd:r:"
#endif

// Global variables to record what we should do in this run.
static int sIterations = 1;
static bool verbose = false;
static bool reuseInput = true;
static bool measureExecTime = false;
static vector<int64_t> dimKnownAtRuntime;

void usage(const char *name) {
#if LOAD_MODEL_STATICALLY
  cout << "Usage: " << name << " [options]";
#else
  cout << "Usage: " << name << " [options] model.so";
#endif
  cout << endl << endl;
  cout << "  Program will instantiate the model given by \"model.so\"" << endl;
  cout << "  with random inputs, launch the computation, and ignore" << endl;
  cout << "  the results. A model is typically generated by lowering" << endl;
  cout << "  an ONNX model using a \"onnx-mlir --EmitLib model.onnx\"" << endl;
  cout << "  command. When the input model is not found as is, the" << endl;
  cout << "  path to the local directory is also prepended." << endl;
  cout << endl;
  cout << "  Options:" << endl;
  cout << "    -d | -dim json-array" << endl;
  cout << "         Provide a json array to provide the value of every" << endl;
  cout << "         runtime dimensions in the input signature of the" << endl;
  cout << "         entry function. E.g. -d \"[7 2048]\"." << endl;
#if !LOAD_MODEL_STATICALLY
  cout << "    -e name | --entry-point name" << endl;
  cout << "         Name of the ONNX model entry point." << endl;
  cout << "         Default is \"run_main_graph\"." << endl;
#endif
  cout << "    -h | --help" << endl;
  cout << "         Print help message." << endl;
  cout << "    -n NUM | --iterations NUM" << endl;
  cout << "         Number of times to run the tests, default 1." << endl;
  cout << "    -m NUM | --meas NUM" << endl;
  cout << "         Measure the kernel execution time NUM times." << endl;
  cout << "    -r | -reuse true|false" << endl;
  cout << "         Reuse input data, default on" << endl;
  cout << "    -v | --verbose" << endl;
  cout << "         Print the shape of the inputs and outputs." << endl;
  cout << endl;
  exit(1);
}

void loadDLL(string name, string entryPointName) {
  cout << "Load model file " << name << " with entry point " << entryPointName
       << endl;
  void *handle = dlopen(name.c_str(), RTLD_LAZY);
  if (!handle) {
    string qualifiedName = "./" + name;
    cout << "  Did not find model, try in current dir " << qualifiedName
         << endl;
    handle = dlopen(qualifiedName.c_str(), RTLD_LAZY);
  }
  assert(handle && "Error loading the model's dll file; you may have provide a "
                   "fully qualified path");
  dll_run_main_graph = (OMTensorList * (*)(OMTensorList *))
      dlsym(handle, entryPointName.c_str());
  assert(!dlerror() && "failed to load entry point");
  dll_omInputSignature = (const char *(*)())dlsym(handle, "omInputSignature");
  assert(!dlerror() && "failed to load omInputSignature");
  dll_omOutputSignature = (const char *(*)())dlsym(handle, "omOutputSignature");
  assert(!dlerror() && "failed to load omOutputSignature");
  dll_omTensorCreate =
      (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE))
          dlsym(handle, "omTensorCreate");
  assert(!dlerror() && "failed to load omTensorCreate");
  dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int))
      dlsym(handle, "omTensorListCreate");
  assert(!dlerror() && "failed to load omTensorListCreate");
  dll_omTensorListDestroy =
      (void (*)(OMTensorList *))dlsym(handle, "omTensorListDestroy");
  assert(!dlerror() && "failed to load omTensorListDestroy");
}

// Parse input arguments.
void parseArgs(int argc, char **argv) {
  dimKnownAtRuntime.clear();
  int c;
  string entryPointName("run_main_graph");
  static struct option long_options[] = {
      {"dim", required_argument, 0, 'd'},         // dimensions.
      {"entry-point", required_argument, 0, 'e'}, // Entry point.
      {"help", no_argument, 0, 'h'},              // Help.
      {"iterations", required_argument, 0, 'n'},  // Number of iterations.
      {"meas", required_argument, 0, 'm'},        // Measurement of time.
      {"reuse", required_argument, 0, 'r'},       // cached input.
      {"verbose", no_argument, 0, 'v'},           // Verbose.
      {0, 0, 0, 0}};

  while (true) {
    int index = 0;
    c = getopt_long(argc, argv, OPTIONS, long_options, &index);
    if (c == -1)
      break;
    switch (c) {
    case 0:
      break;
    case 'd': {
      // Read json array for undefined values.
      dimKnownAtRuntime.clear();
      auto JSONInput = llvm::json::parse(optarg);
      assert(JSONInput && "failed to parse json");
      auto JSONArray = JSONInput->getAsArray();
      assert(JSONArray && "failed to parse json as array");
      int inputNum = JSONArray->size();
      for (int i = 0; i < inputNum; ++i) {
        auto JSONDimValue = (*JSONArray)[i].getAsInteger();
        assert(JSONDimValue && "failed to get value");
        int64_t dim = JSONDimValue.getValue();
        dimKnownAtRuntime.push_back(dim);
      }
      break;
    }
    case 'e':
      entryPointName = optarg;
      break;
    case 'n':
      sIterations = atoi(optarg);
      break;
    case 'm':
#ifdef _WIN32
      cout << "> Measurement option currently not available, ignore." << endl;
      break;
#endif
      sIterations = atoi(optarg);
      measureExecTime = true;
      break;
    case 'r':
      if (strcmp(optarg, "true") == 0) {
        reuseInput = true;
        printf("> Reuse input data\n");
      } else if (strcmp(optarg, "false") == 0) {
        reuseInput = false;
        printf("> Do not reuse input data\n");
      } else {
        printf("  reuse parameter expect true/false argument\n");
        usage(argv[0]);
      }
      break;
    case 'v':
      verbose = true;
      break;
    default:
      usage(argv[0]);
    }
  }
  // Make sure that iterations are positive.
  if (sIterations < 1)
    sIterations = 1;

// Process the DLL.
#if LOAD_MODEL_STATICALLY
  if (optind < argc) {
    cout << "Error: model.so was compiled in, cannot provide one now" << endl;
    usage(argv[0]);
  }
#else
  if (optind == argc) {
    cout << "Error: need one model.so dynamic library" << endl;
    usage(argv[0]);
  } else if (optind + 1 == argc) {
    string name = argv[optind];
    loadDLL(name, entryPointName);
  } else {
    cout << "Error: handle only one model.so dynamic library at a time" << endl;
    usage(argv[0]);
  }
#endif
}

/**
 * \brief Create and initialize an OMTensorList from the signature of a model
 *
 * This function parse the signature of a ONNX compiled network, attached to the
 * binary via a .so and will scan the JSON signature for its input. For each
 * input in turn, it create a tensor of the proper type and shape. Data will be
 * either initialized (if dataPtrList is provided), allocated (if dataPtrList is
 * null and dataAlloc is set to true), or will otherwise be left empty. In case
 * of errors, a null pointer returned.
 *
 *
 * @param dataPtrList Pointer to a list of data pointers of the right size, as
 * determined by the signature.
 * @param allocData When no dataPtrList is provided, the this boolean variable
 * determine if data is to be allocated or not, using the sizes determined by
 * the signature.
 * @param trace If true, provide a printout of the signatures (input and
 * putput).
 * @return pointer to the TensorList just created, or null on error.
 */
OMTensorList *omTensorListCreateFromInputSignature(
    void **dataPtrList, bool dataAlloc, bool trace, bool silent) {
  const char *sigIn = OM_INPUT_SIGNATURE();
  if (trace) {
    cout << "Model Input Signature " << (sigIn ? sigIn : "(empty)") << endl;
    const char *sigOut = OM_OUTPUT_SIGNATURE();
    cout << "Output signature: " << (sigOut ? sigOut : "(empty)") << endl;
  }
  if (!sigIn)
    return nullptr;

  // Create inputs.
  auto JSONInput = llvm::json::parse(sigIn);
  assert(JSONInput && "failed to parse json");
  auto JSONArray = JSONInput->getAsArray();
  assert(JSONArray && "failed to parse json as array");

  // Allocate array of inputs.
  int inputNum = JSONArray->size();
  assert(inputNum >= 0 && inputNum < 100 && "out of bound number of inputs");
  OMTensor **inputTensors = nullptr;
  if (inputNum > 0)
    inputTensors = (OMTensor **)malloc(inputNum * sizeof(OMTensor *));
  // Scan each input tensor
  int dimKnownAtRuntimeIndex = 0;
  for (int i = 0; i < inputNum; ++i) {
    auto JSONItem = (*JSONArray)[i].getAsObject();
    auto JSONItemType = JSONItem->getString("type");
    assert(JSONItemType && "failed to get type");
    auto type = JSONItemType.getValue();
    auto JSONDimArray = JSONItem->getArray("dims");
    int rank = JSONDimArray->size();
    assert(rank > 0 && rank < 100 && "rank is out bound");
    // Gather shape.
    int64_t shape[100];
    size_t size = 1;
    for (int d = 0; d < rank; ++d) {
      auto JSONDimValue = (*JSONDimArray)[d].getAsInteger();
      assert(JSONDimValue && "failed to get value");
      int64_t dim = JSONDimValue.getValue();
      if (dim < 0) {
        // we have a runtime value
        if (dimKnownAtRuntimeIndex >= dimKnownAtRuntime.size()) {
          printf("Error: there are runtime dim for which we have no value; "
                 "provide values using the -d option\n");
          usage("run-onnx-lib");
        }
        dim = dimKnownAtRuntime[dimKnownAtRuntimeIndex++];
        if (!silent || verbose) {
          printf("  Tensor %d, dim %d: use provided RT value %lld\n", i, d,
              (long long)dim);
        }
      }
      shape[d] = dim;
      size *= dim;
    }
    // Create a randomly initialized tensor of the right shape.
    OMTensor *tensor = nullptr;
    if (type.equals("float") || type.equals("f32") || type.equals("i32")) {
      // Treat floats/f32 and i32 alike as they take the same memory footprint.
      float *data = nullptr;
      if (dataPtrList) {
        data = (float *)dataPtrList[i];
      } else if (dataAlloc) {
        data = new float[size];
        assert(data && "failed to allocate data");
      }
      tensor = OM_TENSOR_CREATE(data, shape, rank, ONNX_TYPE_FLOAT);
    } else if (type.equals("double") || type.equals("f64") || type.equals("i64")) {
      // Treat floats/f64 and i64 alike as they take the same memory footprint.
      double *data = nullptr;
      if (dataPtrList) {
        data = (double *)dataPtrList[i];
      } else if (dataAlloc) {
        data = new double[size];
        assert(data && "failed to allocate data");
      }
      tensor = OM_TENSOR_CREATE(data, shape, rank, ONNX_TYPE_DOUBLE);
    }

    assert(tensor && "add support for the desired type");
    // Add tensor to list.
    inputTensors[i] = tensor;
    if (trace) {
      cout << "Input " << i << ": tensor of " << type.str() << " with shape ";
      for (int d = 0; d < rank; ++d)
        cout << shape[d] << " ";
      cout << "and " << size << " elements" << endl;
    }
  }
  return OM_TENSOR_LIST_CREATE(inputTensors, inputNum);
}

// Data structures for timing info.
#ifdef _WIN32
#else
struct timeval startTime, stopTime, result;
#endif

// Process start time.
#ifdef _WIN32
inline void processStartTime() {}
#else
inline void processStartTime() {
  if (!measureExecTime)
    return;
  gettimeofday(&startTime, NULL);
}
#endif

// Process stop time, store result in log.
#ifdef _WIN32
inline void processStopTime() {}
#else
inline void processStopTime() {
  if (!measureExecTime)
    return;
  gettimeofday(&stopTime, NULL);
  timersub(&stopTime, &startTime, &result);
  uint64_t time =
      (uint64_t)result.tv_sec * 1000000ull + (uint64_t)result.tv_usec;
  timeLogInMicroSec.emplace_back(time);
}
#endif

// Print timing info, removing shortest/longest measured time.
void printTime(double avg, double std, double factor, string unit) {
  int s = timeLogInMicroSec.size();
  int m = s / 2;
  printf("@time, %s, median, %.1f, avg, %.1f, std, %.1f, min, %.1f, max, %.1f, "
         "sample, %d\n",
      unit.c_str(), (double)timeLogInMicroSec[m] / factor,
      (double)(avg / factor), (double)(std / factor),
      (double)timeLogInMicroSec[0] / factor,
      (double)timeLogInMicroSec[s - 1] / factor, s);
}

void displayTime() {
  int s = timeLogInMicroSec.size();
  if (s == 0)
    return;
  sort(timeLogInMicroSec.begin(), timeLogInMicroSec.end());
  double avg = 0;
  for (int i = 0; i < s; ++i)
    avg += (double)timeLogInMicroSec[i];
  avg = avg / s;
  double std = 0;
  for (int i = 0; i < s; ++i)
    std += ((double)timeLogInMicroSec[i] - avg) *
           ((double)timeLogInMicroSec[i] - avg);
  std = sqrt(std / s);
  printTime(avg, std, 1, "micro-second");
  if (avg >= 1e3) {
    printTime(avg, std, 1e3, "milli-second");
  }
  if (avg >= 1e6) {
    printTime(avg, std, 1e6, "second");
  }
}

// Perform generation of input, run, measure time,...
int main(int argc, char **argv) {
  // Init args.
  parseArgs(argc, argv);
  // Init inputs.
  OMTensorList *tensorListIn =
      omTensorListCreateFromInputSignature(nullptr, true, verbose, false);
  assert(tensorListIn && "failed to scan signature");
  // Call the compiled onnx model function.
  cout << "Start computing " << sIterations << " iterations" << endl;
  for (int i = 0; i < sIterations; ++i) {
    OMTensorList *tensorListOut = nullptr;
    processStartTime();
    tensorListOut = RUN_MAIN_GRAPH(tensorListIn);
    processStopTime();
    if (tensorListOut)
      OM_TENSOR_LIST_DESTROY(tensorListOut);
    if (i > 0 && i % 10 == 0)
      cout << "  computed " << i << " iterations" << endl;
    if (!reuseInput) {
      OM_TENSOR_LIST_DESTROY(tensorListIn);
      tensorListIn =
          omTensorListCreateFromInputSignature(nullptr, true, false, true);
    }
  }
  cout << "Finish computing " << sIterations << " iterations" << endl;
  displayTime();

  // Cleanup.
  OM_TENSOR_LIST_DESTROY(tensorListIn);
  return 0;
}
