/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ OMTensorListHelper.cpp - OMTensorList C++ debug helpers -------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// Implementation of omTensorListCreateFromInputSignature and its parsing
// helpers. Compiled separately into OMDebugRuntime so it can be linked
// alongside a statically compiled model without conflicting with the model's
// bundled cruntime C symbols.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/Runtime/OMTensorHelper.hpp"
#include "src/Runtime/OMTensorListHelper.hpp"

// =============================================================================
// Support to parse input entries

// Split a string by a single character delimiter.
static std::vector<std::string> splitByChar(
    const std::string &s, char delimiter) {
  std::vector<std::string> result;
  std::istringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delimiter))
    result.push_back(item);
  return result;
}

// Print "Error: <msg>" to stderr and return false.
static bool reportFailure(const std::string &msg) {
  fprintf(stderr, "Error: %s\n", msg.c_str());
  return false;
}

// Parse a pre-split entry string: "num1[-num2]:content".
// Valid index patterns:
//   -1:content    : all inputs (idLow = idHigh = -1)
//   N:content     : single input N (0 <= N < numInputs)
//   N-M:content   : range from N to M (0 <= N <= M < numInputs)
static bool parseEntry(const std::string &entry, int numInputs, int64_t &idLow,
    int64_t &idHigh, std::string &content) {
  auto parts = splitByChar(entry, ':');
  if (parts.size() != 2)
    return reportFailure(
        "Expected one ':' separator in '" + entry + "'");
  const std::string &indexStr = parts[0];
  content = parts[1];

  std::istringstream iss(indexStr);
  iss >> std::ws;
  if (!(iss >> idLow))
    return reportFailure("Failed to parse input ID from '" + entry + "'");

  if (idLow != -1 && (idLow < 0 || idLow >= numInputs))
    return reportFailure("Input ID " + std::to_string(idLow) +
                         " out of range (must be -1 or 0.." +
                         std::to_string(numInputs - 1) + ")");
  idHigh = idLow;
  if (idLow >= 0) {
    iss >> std::ws;
    if (iss.peek() == '-') {
      iss.get();
      if (!(iss >> idHigh))
        return reportFailure(
            "Failed to parse range end from '" + entry + "'");
      if (idHigh < idLow || idHigh >= numInputs)
        return reportFailure("Range end " + std::to_string(idHigh) +
                             " invalid (must be " + std::to_string(idLow) +
                             ".." + std::to_string(numInputs - 1) + ")");
    }
  }
  return true;
}

// =============================================================================
// Shape information parsing.

static bool parseShapeInfo(const char *shapeInfo, int numInputs,
    std::vector<std::vector<int64_t>> &overrides) {
  overrides.assign(numInputs, {});
  if (!shapeInfo || shapeInfo[0] == '\0')
    return true;

  for (const auto &entry : splitByChar(std::string(shapeInfo), ',')) {
    int64_t idLow, idHigh;
    std::string content;
    if (!parseEntry(entry, numInputs, idLow, idHigh, content))
      return reportFailure("Failed to parse shapeInfo entry");

    std::vector<int64_t> dims;
    for (const auto &dimStr : splitByChar(content, 'x')) {
      std::istringstream ds(dimStr);
      ds >> std::ws;
      int64_t d;
      if (!(ds >> d))
        return reportFailure(
            "Failed to parse dimension from '" + dimStr + "'");
      if (d < -1)
        return reportFailure(
            "expected dimension value: -1 or positive, got " +
            std::to_string(d));
      dims.push_back(d);
    }
    if (idLow == -1) {
      for (int i = 0; i < numInputs; ++i)
        overrides[i] = dims;
    } else {
      for (int64_t i = idLow; i <= idHigh; ++i)
        overrides[(int)i] = dims;
    }
  }
  return true;
}

// =============================================================================
// Value information parsing.

struct ValueSpec {
  double minVal = 0.0;
  double maxVal = 0.0;
  bool hasMin = false;
  bool hasMax = false;
};

static bool parseValueInfo(
    const char *valueInfo, int numInputs, std::vector<ValueSpec> &specs) {
  specs.assign(numInputs, ValueSpec{});
  if (!valueInfo || valueInfo[0] == '\0')
    return true;

  for (const auto &entry : splitByChar(std::string(valueInfo), ',')) {
    int64_t idLow, idHigh;
    std::string content;
    if (!parseEntry(entry, numInputs, idLow, idHigh, content))
      return reportFailure("Failed to parse valueInfo entry");

    ValueSpec vspec;
    std::istringstream specStream(content);
    std::string token;
    while (specStream >> token) {
      if (token.size() < 4)
        return reportFailure(
            "Unrecognized value spec token '" + token + "'");
      std::string key = token.substr(0, 3);
      std::istringstream valStream(token.substr(3));
      double v;
      if (!(valStream >> v))
        return reportFailure(
            "Failed to parse number from '" + token + "'");
      if (key == "min") {
        vspec.minVal = v;
        vspec.hasMin = true;
      } else if (key == "max") {
        vspec.maxVal = v;
        vspec.hasMax = true;
      } else if (key == "val") {
        vspec.minVal = vspec.maxVal = v;
        vspec.hasMin = vspec.hasMax = true;
      } else {
        return reportFailure(
            "Unrecognized value spec keyword '" + key + "'");
      }
    }
    if (idLow == -1) {
      for (int i = 0; i < numInputs; ++i)
        specs[i] = vspec;
    } else {
      for (int64_t i = idLow; i <= idHigh && i < numInputs; ++i)
        specs[(int)i] = vspec;
    }
  }
  return true;
}

// =============================================================================
// JSON input-signature parsing.

static void skipWS(std::string_view &s) {
  size_t i = s.find_first_not_of(" \t\n\r");
  s = (i == std::string_view::npos) ? s.substr(s.size()) : s.substr(i);
}

static bool consume(std::string_view &s, char c) {
  skipWS(s);
  if (s.empty() || s[0] != c)
    return false;
  s = s.substr(1);
  return true;
}

static bool parseJsonStr(std::string_view &s, std::string &out) {
  skipWS(s);
  if (s.empty() || s[0] != '"')
    return false;
  s = s.substr(1);
  size_t end = s.find('"');
  if (end == std::string_view::npos)
    return false;
  out.assign(s.data(), end);
  s = s.substr(end + 1);
  return true;
}

static bool parseJsonInt(std::string_view &s, int64_t &val) {
  skipWS(s);
  if (s.empty())
    return false;
  char *end;
  val = (int64_t)strtoll(s.data(), &end, 10);
  if (end == s.data())
    return false;
  s = s.substr((size_t)(end - s.data()));
  return true;
}

// MLIR type name array generated from metadata.
static const char *OM_DATA_TYPE_MLIR_NAME[] = {
#define OM_TYPE_METADATA_DEF(                                                  \
    ENUM_NAME, ENUM_VAL, DTYPE_SIZE, DTYPE_NAME, MLIR_NAME, NUMPY_NAME)        \
  MLIR_NAME,
#include "onnx-mlir/Runtime/OnnxDataTypeMetaData.inc"
#undef OM_TYPE_METADATA_DEF
};

// NumPy type name array generated from metadata.
static const char *OM_DATA_TYPE_NUMPY_NAME[] = {
#define OM_TYPE_METADATA_DEF(                                                  \
    ENUM_NAME, ENUM_VAL, DTYPE_SIZE, DTYPE_NAME, MLIR_NAME, NUMPY_NAME)        \
  NUMPY_NAME,
#include "onnx-mlir/Runtime/OnnxDataTypeMetaData.inc"
#undef OM_TYPE_METADATA_DEF
};

static OM_DATA_TYPE typeStringToOMDataType(const std::string &typeStr) {
  if (typeStr == "float")
    return ONNX_TYPE_FLOAT;
  if (typeStr == "double")
    return ONNX_TYPE_DOUBLE;
  for (int i = 0; i <= ONNX_TYPE_LAST; ++i) {
    if (typeStr == OM_DATA_TYPE_MLIR_NAME[i])
      return (OM_DATA_TYPE)i;
  }
  return ONNX_TYPE_UNDEFINED;
}

struct SigEntry {
  OM_DATA_TYPE type;
  std::vector<int64_t> dims;
};

static bool parseSignatureInfo(
    const char *inputSignatureStr, std::vector<SigEntry> &entries) {
  if (!inputSignatureStr)
    return false;

  std::string_view s(inputSignatureStr);
  if (!consume(s, '['))
    return false;

  while (true) {
    skipWS(s);
    if (s.empty() || s[0] == ']')
      break;
    if (s[0] == ',') { s = s.substr(1); continue; }
    if (!consume(s, '{'))
      return false;

    SigEntry entry;
    while (true) {
      skipWS(s);
      if (s.empty())
        return false;
      if (s[0] == '}') { s = s.substr(1); break; }
      if (s[0] == ',') { s = s.substr(1); continue; }

      std::string key;
      if (!parseJsonStr(s, key) || !consume(s, ':'))
        return false;

      if (key == "type") {
        std::string val;
        if (!parseJsonStr(s, val))
          return false;
        entry.type = typeStringToOMDataType(val);
        if (entry.type == ONNX_TYPE_UNDEFINED) {
          fprintf(stderr, "parseSignatureInfo: unsupported type '%s'\n",
              val.c_str());
          return false;
        }
      } else if (key == "dims") {
        if (!consume(s, '['))
          return false;
        while (true) {
          skipWS(s);
          if (s.empty())
            return false;
          if (s[0] == ']') { s = s.substr(1); break; }
          if (s[0] == ',') { s = s.substr(1); continue; }
          int64_t dim;
          if (!parseJsonInt(s, dim))
            return false;
          entry.dims.push_back(dim);
        }
      } else {
        std::string tmp;
        parseJsonStr(s, tmp);
      }
    }
    entries.push_back(std::move(entry));
  }
  return !entries.empty();
}

static const char *omDataTypeToString(OM_DATA_TYPE type) {
  if (type >= 0 && type <= ONNX_TYPE_LAST)
    return OM_DATA_TYPE_MLIR_NAME[type];
  return "unknown";
}

static bool parseInputSpecs(const char *inputSignatureStr,
    const char *shapeInfo, std::vector<SigEntry> &entries) {
  if (!parseSignatureInfo(inputSignatureStr, entries))
    return false;

  int inputNum = (int)entries.size();
  std::vector<std::vector<int64_t>> shapeOverrides;
  if (!parseShapeInfo(shapeInfo, inputNum, shapeOverrides))
    return false;

  for (int i = 0; i < inputNum; ++i) {
    if (!shapeOverrides[i].empty()) {
      const std::vector<int64_t> &over = shapeOverrides[i];
      std::vector<int64_t> &shape = entries[i].dims;
      for (int d = 0; d < (int)shape.size() && d < (int)over.size(); ++d) {
        int64_t sigDim = shape[d];
        int64_t overDim = over[d];
        if (sigDim >= 0 && overDim >= 0) {
          if (sigDim != overDim)
            return reportFailure(
                "parseInputSpecs: tensor " + std::to_string(i) +
                " dim " + std::to_string(d) + ": signature value " +
                std::to_string(sigDim) + " conflicts with shapeInfo value " +
                std::to_string(overDim));
        } else if (sigDim < 0 && overDim >= 0) {
          shape[d] = overDim;
        } else if (sigDim >= 0) {
          // keep sigDim
        } else {
          return reportFailure(
              "parseInputSpecs: tensor " + std::to_string(i) +
              " dim " + std::to_string(d) +
              " is dynamic in both signature and shapeInfo; "
              "provide a static value");
        }
      }
    }
  }
  return true;
}

// =============================================================================
// Default random bounds per type.

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
    return 0.0;
  case ONNX_TYPE_STRING:
    return 0.0;  // strings: random integers in [0, 64)
  default:
    return 0.0;
  }
}

static double defaultRandUB(OM_DATA_TYPE t) {
  switch (t) {
  case ONNX_TYPE_FLOAT:
  case ONNX_TYPE_DOUBLE:
  case ONNX_TYPE_FLOAT16:
    return 0.1;
  case ONNX_TYPE_BOOL:
    return 1.0;
  case ONNX_TYPE_STRING:
    return 63.0;  // strings: random integers in [0, 64)
  default:
    return 10.0;
  }
}

static OM_DATA_TYPE runONNXModelNameToOMType(const std::string &name) {
  for (int i = 0; i <= ONNX_TYPE_LAST; ++i) {
    if (OM_DATA_TYPE_NUMPY_NAME[i][0] != '\0' &&
        name == OM_DATA_TYPE_NUMPY_NAME[i])
      return (OM_DATA_TYPE)i;
  }
  // "bool" is accepted as an alias for "bool_" (the canonical numpy name).
  if (name == "bool")
    return ONNX_TYPE_BOOL;
  return ONNX_TYPE_UNDEFINED;
}

static bool parseBoundOverrides(const char *bounds, double *arr) {
  if (!bounds || bounds[0] == '\0')
    return true;

  for (const auto &entry : splitByChar(std::string(bounds), ',')) {
    auto parts = splitByChar(entry, ':');
    if (parts.size() != 2)
      return reportFailure(
          "Expected one ':' separator in bound entry '" + entry + "'");
    std::istringstream typeStream(parts[0]);
    std::string typeName;
    if (!(typeStream >> typeName))
      return reportFailure(
          "Empty type name in bound entry '" + entry + "'");
    std::istringstream valStream(parts[1]);
    double val;
    if (!(valStream >> val))
      return reportFailure(
          "Failed to parse bound value from '" + parts[1] + "'");
    OM_DATA_TYPE t = runONNXModelNameToOMType(typeName);
    if (t == ONNX_TYPE_UNDEFINED)
      return reportFailure(
          "Unknown type name '" + typeName + "' in bound override");
    arr[t] = val;
  }
  return true;
}

// =============================================================================
// Public entry point.

OMTensorList *omTensorListCreateFromInputSignature(
    const char *inputSignatureStr, const char *shapeInfo,
    const char *valueInfo, const char *defaultLowerBound,
    const char *defaultUpperBound, bool verbose) {

  std::vector<SigEntry> entries;
  if (!parseInputSpecs(inputSignatureStr, shapeInfo, entries))
    return nullptr;

  int inputNum = (int)entries.size();
  std::vector<ValueSpec> valueSpecs;
  if (!parseValueInfo(valueInfo, inputNum, valueSpecs))
    return nullptr;

  double lb[ONNX_TYPE_LAST + 1], ub[ONNX_TYPE_LAST + 1];
  for (int t = 0; t <= ONNX_TYPE_LAST; ++t) {
    lb[t] = defaultRandLB((OM_DATA_TYPE)t);
    ub[t] = defaultRandUB((OM_DATA_TYPE)t);
  }
  if (!parseBoundOverrides(defaultLowerBound, lb))
    return nullptr;
  if (!parseBoundOverrides(defaultUpperBound, ub))
    return nullptr;

  OMTensor **inputTensors =
      (OMTensor **)malloc((size_t)inputNum * sizeof(OMTensor *));
  if (!inputTensors)
    return nullptr;

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
    double lo = vspec.hasMin ? vspec.minVal : lb[entry.type];
    double hi = vspec.hasMax ? vspec.maxVal : ub[entry.type];
    tensor = omTensorCreateWithRandomData(entry.dims, entry.type, lo, hi);
    if (!tensor)
      return abortWithCleanup(i);
    inputTensors[i] = tensor;
    if (verbose) {
      printf("Input %d: tensor of %s with shape", i,
          omDataTypeToString(entry.type));
      for (int d = 0; d < rank; ++d)
        printf(" %lld", (long long)entry.dims[d]);
      if (entry.type == ONNX_TYPE_STRING)
        printf(", value range [\"%g\", \"%g\"]\n", lo, hi);
      else
        printf(", value range [%g, %g]\n", lo, hi);
    }
  }

  OMTensorList *list = omTensorListCreate(inputTensors, inputNum);
  free(inputTensors);
  return list;
}
