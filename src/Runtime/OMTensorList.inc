/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMTensorList.inc - OMTensor C/C++ Implementation ----------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMTensorList data
// structures and helper functions.
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#include "onnx-mlir/Runtime/OMTensorList.h"
#include <string.h>

#define OM_DRIVER_TIMING 1
#include "src/Runtime/OMInstrumentHelper.h"

struct OMTensorList {
#ifdef __cplusplus

  /**
   * Constructor
   *
   * Create an OMTensorList with specified OMTensor pointer array, the size of
   * the array and the array and indication of whether the destructor should
   * use OMTensorListDestroyShallow
   */
  OMTensorList(OMTensor *omts[], int64_t n, bool shallow = false)
      : _omts(omts), _size(n), _shallow(shallow){};

  /**
   * Destructor
   *
   * Destroy the OMTensorList struct.
   */
  ~OMTensorList() {
    /* Destroy all the OMTensors */
    if (!_shallow)
      for (int64_t i = 0; i < _size; i++)
        omTensorDestroy(_omts[i]);
    free(_omts);
  };
#endif

  /* To facilitate user facing API getOmts, OMTensors are kept in a vector
   * that can be quickly returned as an array. A name to index map is used
   * to address ReMemRefs by name.
   */
  OMTensor **_omts; // OMTensor array

  int64_t _size; // Number of elements in _omts.

  bool _shallow; // Internal boolean for the C++ constructor/destructor API
                 // This indicates whether to perform a shallow destroy.
};

/* OMTensorList creator */
OMTensorList *omTensorListCreate(OMTensor **tensors, int64_t n) {
  OMTensorList *list = (OMTensorList *)malloc(sizeof(struct OMTensorList));
  if (!list)
    return NULL;

  TIMING_INIT_START(tensor_list_create)
  list->_size = n;
  if (n > 0) {
    size_t omts_bytes = sizeof(OMTensor *) * n;
    list->_omts = (OMTensor **)malloc(omts_bytes);

    if (!list->_omts) {
      free(list); // free the previously allocated memory in case of an error
      return NULL;
    }
    // Copy the given OMTensors pointers to an array owned by OMTensorList
    memcpy(list->_omts, tensors, omts_bytes);
  } else {
    list->_omts = NULL;
  }
  TIMING_STOP_PRINT(tensor_list_create);
  return list;
}

/* OMTensorList destroyer */
void omTensorListDestroy(OMTensorList *list) {
  if (!list)
    return;
  TIMING_INIT_START(tensor_list_destroy)
  for (int64_t i = 0; i < list->_size; i++)
    omTensorDestroy(list->_omts[i]);
  // Free the list as well as the pointers to the OMTensor array
  omTensorListDestroyShallow(list);
  TIMING_STOP_PRINT(tensor_list_destroy)
}

/* OMTensorList destroyer which does not destroy the tensors.
 */
void omTensorListDestroyShallow(OMTensorList *list) {
  if (!list)
    return;
  // Free the list as well as the pointer to the OMTensor array
  if (list->_omts)
    free(list->_omts);
  free(list);
}

/* OMTensorList OMTensor array getter */
OMTensor **omTensorListGetOmtArray(const OMTensorList *list) {
  if (!list)
    return NULL;
  return list->_omts;
}

/* OMTensorList number of OMTensor getter */
int64_t omTensorListGetSize(const OMTensorList *list) {
  if (!list)
    return 0;
  return list->_size;
}

/* Return OMTensor at specified index in the OMTensorList */
OMTensor *omTensorListGetOmtByIndex(const OMTensorList *rlist, int64_t index) {
  if (!rlist || index < 0 || index >= rlist->_size)
    return NULL;
  return rlist->_omts[index];
}

#ifdef __cplusplus

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "src/Runtime/OMTensorHelper.hpp"

// =============================================================================
// Support to parse input entries

// Parse an entry from a string: "num1[-num2]<string>(comma or end)".
// Returns true on success, false on error.
// On success:
//   - idLow and idHigh are set to the parsed range
//   - content is set to the string content (after optional ':')
//   - remaining is set to the unparsed portion of the string
// Valid patterns:
//   -1:content        : all inputs (idLow = idHigh = -1)
//   N:content         : single input N (idLow = idHigh = N, 0 <= N < numInputs)
//   N-M:content       : range from N to M (0 <= N <= M < numInputs)
// Invalid patterns return false.
static bool parseEntry(const std::string &input, int numInputs, int64_t &idLow,
    int64_t &idHigh, std::string &content, std::string &remaining) {
  std::istringstream iss(input);
  
  // Skip leading whitespace.
  iss >> std::ws;
  
  // Check for end of string.
  if (iss.eof()) {
    fprintf(stderr, "Error: Empty entry in input specification\n");
    return false;
  }

  // Parse the input ID range.
  if (!(iss >> idLow)) {
    fprintf(stderr, "Error: Failed to parse input ID from '%s'\n",
        input.c_str());
    return false;
  }

  // Validate idLow.
  if (idLow != -1 && (idLow < 0 || idLow >= numInputs)) {
    fprintf(stderr,
        "Error: Input ID %lld out of range (must be -1 or 0..%d)\n",
        (long long)idLow, numInputs - 1);
    return false;
  }

  // Check for range syntax "N-M".
  idHigh = idLow;
  if (idLow >= 0 && iss.peek() == '-') {
    iss.get(); // Consume the '-'
    if (!(iss >> idHigh)) {
      fprintf(stderr, "Error: Failed to parse range end from '%s'\n",
          input.c_str());
      return false;
    }

    // Validate idHigh: must be >= idLow and < numInputs.
    if (idHigh < idLow || idHigh >= numInputs) {
      fprintf(stderr,
          "Error: Range end %lld invalid (must be %lld..%d)\n",
          (long long)idHigh, (long long)idLow, numInputs - 1);
      return false;
    }
  }

  // Skip whitespace and require ':'.
  iss >> std::ws;
  if (iss.peek() != ':') {
    fprintf(stderr, "Error: Missing ':' separator in '%s'\n", input.c_str());
    return false;
  }
  iss.get(); // Consume the ':'

  // Extract content up to comma or end.
  std::getline(iss, content, ',');
  
  // Get remaining string after comma.
  std::getline(iss, remaining);

  return true;
}

// =============================================================================
// Support to parse shape information

// Parse a --shapeInformation-style string into per-input dimension overrides.
// Format: "INPUT_ID:D1xD2x...xDn, ..." where INPUT_ID is an integer >= 0, a
// range "lo-hi", or -1 for all inputs. A dim value of -1 means keep the value
// from the signature. Returns a vector of size numInputs; an empty inner
// vector means no override for that input.
static std::vector<std::vector<int64_t>> parseShapeInfo(
    const char *shapeInfo, int numInputs) {
  std::vector<std::vector<int64_t>> overrides(numInputs);
  if (!shapeInfo || shapeInfo[0] == '\0')
    return overrides;

  std::string input(shapeInfo);
  while (!input.empty()) {
    // Parse the entry: "id[-id]:dims".
    int64_t idLow, idHigh;
    std::string content, remaining;
    if (!parseEntry(input, numInputs, idLow, idHigh, content, remaining)) {
      fprintf(stderr, "Error: Failed to parse shapeInfo entry\n");
      return overrides; // Return empty overrides on error.
    }

    // Parse dimensions from content string, separated by 'x' or 'X'.
    std::vector<int64_t> dims;
    std::istringstream dimStream(content);
    int64_t d;
    while (dimStream >> d) {
      dims.push_back(d);
      // Skip 'x' or 'X' separator.
      dimStream >> std::ws;
      if (dimStream.peek() == 'x' || dimStream.peek() == 'X')
        dimStream.get();
    }

    // Apply dimensions to the specified range.
    if (idLow == -1) {
      for (int i = 0; i < numInputs; ++i)
        overrides[i] = dims;
    } else {
      for (int64_t i = idLow; i <= idHigh; ++i)
        overrides[(int)i] = dims;
    }

    input = remaining;
  }
  return overrides;
}

// =============================================================================
// Support to parse value information

// Per-tensor data fill specification parsed from the valueInfo string.
// Format: "INPUT_ID:spec1spec2..., ..." where each spec is one of:
//   min<num>  lower bound for random fill (default: type-specific, see below)
//   max<num>  upper bound for random fill (default: type-specific, see below)
//   val<num>  constant fill (shorthand for min=max=num)
// String tensors are not supported.
struct ValueSpec {
  double minVal = 0.0;
  double maxVal = 0.0;
  bool hasMin = false; // true if min (or val) was explicitly set
  bool hasMax = false; // true if max (or val) was explicitly set
};

// Parse valueInfo into one ValueSpec per input. Returns false on error.
static bool parseValueInfo(
    const char *valueInfo, int numInputs, std::vector<ValueSpec> &specs) {
  specs.assign(numInputs, ValueSpec{});
  if (!valueInfo || valueInfo[0] == '\0')
    return true;

  std::string input(valueInfo);
  while (!input.empty()) {
    // Parse the entry: "id[-id]:specs".
    int64_t idLow, idHigh;
    std::string content, remaining;
    if (!parseEntry(input, numInputs, idLow, idHigh, content, remaining)) {
      fprintf(stderr, "Error: Failed to parse valueInfo entry\n");
      return false;
    }

    // Parse value specs from content string.
    ValueSpec vspec;
    std::istringstream specStream(content);
    std::string token;
    while (specStream >> token) {
      if (token.substr(0, 3) == "min") {
        std::istringstream valStream(token.substr(3));
        if (!(valStream >> vspec.minVal)) {
          fprintf(stderr, "Error: Failed to parse min value from '%s'\n",
              token.c_str());
          return false;
        }
        vspec.hasMin = true;
      } else if (token.substr(0, 3) == "max") {
        std::istringstream valStream(token.substr(3));
        if (!(valStream >> vspec.maxVal)) {
          fprintf(stderr, "Error: Failed to parse max value from '%s'\n",
              token.c_str());
          return false;
        }
        vspec.hasMax = true;
      } else if (token.substr(0, 3) == "val") {
        std::istringstream valStream(token.substr(3));
        double v;
        if (!(valStream >> v)) {
          fprintf(stderr, "Error: Failed to parse val value from '%s'\n",
              token.c_str());
          return false;
        }
        vspec.minVal = vspec.maxVal = v;
        vspec.hasMin = vspec.hasMax = true;
      } else {
        fprintf(stderr, "Error: Unrecognized value spec token '%s'\n",
            token.c_str());
        return false;
      }
    }

    // Apply value spec to the specified range.
    if (idLow == -1) {
      for (int i = 0; i < numInputs; ++i)
        specs[i] = vspec;
    } else {
      for (int64_t i = idLow; i <= idHigh && i < numInputs; ++i)
        specs[(int)i] = vspec;
    }

    input = remaining;
  }
  return true;
}

// =============================================================================
// Support to parse JSON input function signature.

// JSON parsing helpers for the ONNX signature format:
//   [ { "type" : "f32" , "dims" : [1, 28, 28] , "name" : "image" } , ... ]
// These are file-local helpers used only by
// omTensorListCreateFromInputSignature.

static const char *skipJsonWS(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    ++p;
  return p;
}

// Parse a JSON quoted string into buf (at most bufLen-1 chars). Returns the
// position after the closing quote, or p unchanged if no opening quote found.
static const char *parseJsonString(const char *p, char *buf, size_t bufLen) {
  p = skipJsonWS(p);
  if (*p != '"') {
    buf[0] = '\0';
    return p;
  }
  ++p;
  size_t i = 0;
  while (*p && *p != '"' && i < bufLen - 1)
    buf[i++] = *p++;
  buf[i] = '\0';
  if (*p == '"')
    ++p;
  return p;
}

// Parse a decimal integer (possibly negative). Returns position after the
// number.
static const char *parseJsonInt(const char *p, int64_t *val) {
  p = skipJsonWS(p);
  char *end;
  *val = (int64_t)strtoll(p, &end, 10);
  return end;
}

// MLIR type name array generated from metadata.
static const char *OM_DATA_TYPE_MLIR_NAME[] = {
#define OM_TYPE_METADATA_DEF(                                                   \
    ENUM_NAME, ENUM_VAL, DTYPE_SIZE, DTYPE_NAME, MLIR_NAME)                    \
  MLIR_NAME,
#include "onnx-mlir/Runtime/OnnxDataTypeMetaData.inc"

#undef OM_TYPE_METADATA_DEF
};

// Map an ONNX-MLIR signature type string to OM_DATA_TYPE.
static OM_DATA_TYPE typeStringToOMDataType(const char *typeStr) {
  // Handle synonyms "float" and "double" as special cases.
  if (strcmp(typeStr, "float") == 0)
    return ONNX_TYPE_FLOAT;
  if (strcmp(typeStr, "double") == 0)
    return ONNX_TYPE_DOUBLE;

  // Search through the MLIR type name array.
  for (int i = 0; i <= ONNX_TYPE_LAST; ++i) {
    if (strcmp(typeStr, OM_DATA_TYPE_MLIR_NAME[i]) == 0)
      return (OM_DATA_TYPE)i;
  }
  return ONNX_TYPE_UNDEFINED;
}

// Raw per-tensor data extracted directly from the JSON signature string.
struct SigEntry {
  OM_DATA_TYPE type;
  std::vector<int64_t> dims;
};

// Walk the JSON input signature string and fill one SigEntry per tensor.
// Returns false if the string is null, malformed, or empty.
static bool parseSignatureInfo(
    const char *inputSignatureStr, std::vector<SigEntry> &entries) {
  if (!inputSignatureStr)
    return false;

  const char *p = skipJsonWS(inputSignatureStr);
  if (*p != '[')
    return false;
  ++p;

  while (true) {
    p = skipJsonWS(p);
    if (*p == ']' || *p == '\0')
      break;
    if (*p == ',') {
      ++p;
      continue;
    }
    if (*p != '{')
      return false;
    ++p;

    SigEntry entry;
    while (true) {
      p = skipJsonWS(p);
      if (*p == '}') {
        ++p;
        break;
      }
      if (*p == ',') {
        ++p;
        continue;
      }
      char key[64];
      p = parseJsonString(p, key, sizeof(key));
      p = skipJsonWS(p);
      if (*p == ':')
        ++p;
      p = skipJsonWS(p);
      if (strcmp(key, "type") == 0) {
        char val[64];
        p = parseJsonString(p, val, sizeof(val));
        entry.type = typeStringToOMDataType(val);
        if (entry.type == ONNX_TYPE_UNDEFINED) {
          fprintf(stderr,
              "parseSignatureInfo: unsupported type '%s'\n", val);
          return false;
        }
      } else if (strcmp(key, "dims") == 0) {
        if (*p != '[')
          return false;
        ++p;
        while (true) {
          p = skipJsonWS(p);
          if (*p == ']') {
            ++p;
            break;
          }
          if (*p == ',') {
            ++p;
            continue;
          }
          int64_t dim;
          p = parseJsonInt(p, &dim);
          entry.dims.push_back(dim);
        }
      } else {
        if (*p == '"') { // skip unknown string values (e.g. "name")
          char tmp[256];
          p = parseJsonString(p, tmp, sizeof(tmp));
        }
      }
    }
    entries.push_back(std::move(entry));
  }

  return !entries.empty();
}

// =============================================================================
// Combine shape info and signature info

// Get MLIR type name string from OM_DATA_TYPE for diagnostics.
static const char *omDataTypeToString(OM_DATA_TYPE type) {
  if (type >= 0 && type <= ONNX_TYPE_LAST)
    return OM_DATA_TYPE_MLIR_NAME[type];
  return "unknown";
}

// Parse signature and apply shape overrides directly to SigEntry.
// Returns false and prints a diagnostic on the first error.
static bool parseInputSpecs(const char *inputSignatureStr,
    const char *shapeInfo, std::vector<SigEntry> &entries) {
  if (!parseSignatureInfo(inputSignatureStr, entries))
    return false;

  int inputNum = (int)entries.size();
  // Apply shape overrides to entries.
  std::vector<std::vector<int64_t>> shapeOverrides =
      parseShapeInfo(shapeInfo, inputNum);

  for (int i = 0; i < inputNum; ++i) {
    if (!shapeOverrides[i].empty()) {
      const std::vector<int64_t> &over = shapeOverrides[i];
      std::vector<int64_t> &shape = entries[i].dims;
      for (int d = 0; d < (int)shape.size() && d < (int)over.size(); ++d) {
        int64_t sigDim = shape[d];
        int64_t overDim = over[d];
        if (sigDim >= 0 && overDim >= 0) {
          // Both static: values must agree.
          if (sigDim != overDim) {
            fprintf(stderr,
                "parseInputSpecs: tensor %d dim %d: signature value %lld "
                "conflicts with shapeInfo value %lld\n",
                i, d, (long long)sigDim, (long long)overDim);
            return false;
          }
        } else if (sigDim < 0 && overDim >= 0) {
          shape[d] = overDim; // Signature dynamic, shapeInfo static.
        } else if (sigDim >= 0 /* && overDim < 0 */) {
          // Signature static, shapeInfo dynamic: keep sigDim (no-op).
        } else {
          // Both dynamic: caller must supply a concrete value.
          fprintf(stderr,
              "parseInputSpecs: tensor %d dim %d is dynamic in both signature "
              "and shapeInfo; provide a static value\n",
              i, d);
          return false;
        }
      }
    }
  }
  return true;
}

// =============================================================================
// Support for random lower and upper bound limits.

// Default random lower/upper bounds per type, matching RunONNXModel.py's
// DEFAULT_LB / DEFAULT_UB tables.
static double defaultRandLB(OM_DATA_TYPE t) {
  switch (t) {
  case ONNX_TYPE_FLOAT:
  case ONNX_TYPE_DOUBLE:
  case ONNX_TYPE_FLOAT16:
    return -0.1;
  case ONNX_TYPE_INT8:
  case ONNX_TYPE_INT16:
  case ONNX_TYPE_INT32:
  case ONNX_TYPE_INT64:
    return -10.0;
  case ONNX_TYPE_BOOL:
    return -10.0; // treated as int32 in RunONNXModel.py
  default:
    return 0.0; // uint* and other types
  }
}
static double defaultRandUB(OM_DATA_TYPE t) {
  switch (t) {
  case ONNX_TYPE_FLOAT:
  case ONNX_TYPE_DOUBLE:
  case ONNX_TYPE_FLOAT16:
    return 0.1;
  case ONNX_TYPE_BOOL:
    return 9.0; // treated as int32 in RunONNXModel.py
  default:
    return 10.0; // int* and uint*
  }
}

// Map a RunONNXModel.py type name to OM_DATA_TYPE (reverse of the enum).
static OM_DATA_TYPE runONNXModelNameToOMType(const char *name) {
  if (strcmp(name, "float32") == 0)
    return ONNX_TYPE_FLOAT;
  if (strcmp(name, "float64") == 0)
    return ONNX_TYPE_DOUBLE;
  if (strcmp(name, "float16") == 0)
    return ONNX_TYPE_FLOAT16;
  if (strcmp(name, "int8") == 0)
    return ONNX_TYPE_INT8;
  if (strcmp(name, "int16") == 0)
    return ONNX_TYPE_INT16;
  if (strcmp(name, "int32") == 0)
    return ONNX_TYPE_INT32;
  if (strcmp(name, "int64") == 0)
    return ONNX_TYPE_INT64;
  if (strcmp(name, "uint8") == 0)
    return ONNX_TYPE_UINT8;
  if (strcmp(name, "uint16") == 0)
    return ONNX_TYPE_UINT16;
  if (strcmp(name, "uint32") == 0)
    return ONNX_TYPE_UINT32;
  if (strcmp(name, "uint64") == 0)
    return ONNX_TYPE_UINT64;
  if (strcmp(name, "bool") == 0)
    return ONNX_TYPE_BOOL;
  return ONNX_TYPE_UNDEFINED;
}

// Parse a --lower-bound / --upper-bound style string and write overrides into
// arr[OM_DATA_TYPE]. Format: "type1:value1, type2:value2, ..."
static void parseBoundOverrides(const char *bounds, double *arr) {
  if (!bounds || bounds[0] == '\0')
    return;
  const char *p = bounds;
  while (*p) {
    while (*p == ' ' || *p == '\t')
      ++p;
    if (*p == '\0')
      break;
    const char *colon = strchr(p, ':');
    if (!colon)
      break;
    size_t len = (size_t)(colon - p);
    while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t'))
      --len;
    char typeName[32] = {};
    if (len < sizeof(typeName)) {
      memcpy(typeName, p, len);
    }
    p = colon + 1;
    char *end;
    double val = strtod(p, &end);
    OM_DATA_TYPE t = runONNXModelNameToOMType(typeName);
    if (t != ONNX_TYPE_UNDEFINED)
      arr[t] = val;
    p = end;
    while (*p == ',' || *p == ' ' || *p == '\t')
      ++p;
  }
}

// =============================================================================
// Create list using provided values.

OMTensorList *omTensorListCreateFromInputSignature(
    const char *inputSignatureStr, const char *shapeInfo, const char *valueInfo,
    const char *defaultLowerBound, const char *defaultUpperBound,
    bool verbose) {

  // Parse the input signature of the model we are trying to execute, and
  // complement the dynamic shapes with info from shapeInfo.
  std::vector<SigEntry> entries;
  if (!parseInputSpecs(inputSignatureStr, shapeInfo, entries))
    return nullptr;

  // Parse value info (lower/upper bound, possibly specified for each input
  // tensor).
  int inputNum = (int)entries.size();
  std::vector<ValueSpec> valueSpecs;
  if (!parseValueInfo(valueInfo, inputNum, valueSpecs))
    return nullptr;

  // Parse the default lower and upper bounds for this model, type specific.
  // Build per-type lb/ub arrays: start from built-in defaults, then apply
  // any overrides supplied by the caller. Re-initialised on every call.
  double lb[ONNX_TYPE_LAST + 1], ub[ONNX_TYPE_LAST + 1];
  for (int t = 0; t <= ONNX_TYPE_LAST; ++t) {
    lb[t] = defaultRandLB((OM_DATA_TYPE)t);
    ub[t] = defaultRandUB((OM_DATA_TYPE)t);
  }
  parseBoundOverrides(defaultLowerBound, lb);
  parseBoundOverrides(defaultUpperBound, ub);

  OMTensor **inputTensors =
      (OMTensor **)malloc((size_t)inputNum * sizeof(OMTensor *));
  if (!inputTensors)
    return nullptr;

  // Destroy tensors[0..count-1] and free the array — used on every error path
  // inside the loop so that already-created tensors are not leaked.
  auto abortWithCleanup = [&](int count) -> OMTensorList * {
    for (int k = 0; k < count; ++k)
      omTensorDestroy(inputTensors[k]);
    free(inputTensors);
    return nullptr;
  };

  for (int i = 0; i < inputNum; ++i) {
    const SigEntry &entry = entries[i];
    const ValueSpec &vspec = valueSpecs[i];
    int rank = (int)entry.dims.size();

    OMTensor *tensor = nullptr;
    if ((vspec.hasMin || vspec.hasMax) && entry.type != ONNX_TYPE_STRING) {
      for (int d = 0; d < rank; ++d) {
        if (entry.dims[d] < 0) {
          fprintf(stderr,
              "omTensorListCreateFromInputSignature: tensor %d dim %d is "
              "dynamic; cannot allocate data — provide a concrete value via "
              "shapeInfo.\n",
              i, d);
          return abortWithCleanup(i);
        }
      }
      // Priority: explicit valueInfo > caller override > built-in default.
      double lo = vspec.hasMin ? vspec.minVal : lb[entry.type];
      double hi = vspec.hasMax ? vspec.maxVal : ub[entry.type];
      tensor = omTensorCreateWithRandomData(entry.dims, entry.type, lo, hi);
    } else if (vspec.hasMin || vspec.hasMax) {
      // Data was requested but the type does not support random generation.
      fprintf(stderr,
          "omTensorListCreateFromInputSignature: tensor %d has unsupported "
          "type '%s' for data generation.\n",
          i, omDataTypeToString(entry.type));
      return abortWithCleanup(i);
    } else {
      // No data requested: create a metadata-only tensor with a null buffer.
      tensor = omTensorCreateWithOwnership(
          nullptr, entry.dims.data(), rank, entry.type, /*owning=*/0);
    }
    if (!tensor)
      return abortWithCleanup(i);
    inputTensors[i] = tensor;
    if (verbose) {
      printf("Input %d: tensor of %s with shape", i,
          omDataTypeToString(entry.type));
      for (int d = 0; d < rank; ++d)
        printf(" %lld", (long long)entry.dims[d]);
      printf("\n");
    }
  }

  OMTensorList *list = omTensorListCreate(inputTensors, inputNum);
  free(inputTensors);
  return list;
}

#endif // __cplusplus
