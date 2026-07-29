/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- TestGemmVersionConv.cpp - CVE-2026-63632 regression test ----------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// Regression test for CVE-2026-63632.
//
// The ONNX Gemm 7->6 version-converter adapter (gemm_7_6.h:41) reads
// B_shape[1] without a bounds check.  When a Gemm input has rank < 2 this
// causes a heap-buffer-overflow READ.
//
// ImportFrontendModelArray must return CompilerFailure (not crash) when
// invokeOnnxVersionConverter is true and a Gemm input A or B has rank < 2.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/onnx_pb.h"
SUPPRESS_WARNINGS_POP

using namespace ONNX_NAMESPACE;

static void loadDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::ONNXDialect>();
}

// Build an opset-7 Gemm model where input B has rank 1 (shape [28]).
// This is the shape that triggers the OOB read in gemm_7_6.h:41.
static ModelProto buildGemmRank1BModel() {
  ModelProto model;
  model.set_ir_version(6);

  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(7); // opset 7 triggers the 7->6 downgrade path

  GraphProto *graph = model.mutable_graph();
  graph->set_name("gemm_rank1_b");

  const auto floatType = TensorProto_DataType::TensorProto_DataType_FLOAT;

  // Input A: shape [2, 3] — valid rank-2
  ValueInfoProto *a = graph->add_input();
  a->set_name("A");
  auto *att = a->mutable_type()->mutable_tensor_type();
  att->set_elem_type(floatType);
  att->mutable_shape()->add_dim()->set_dim_value(2);
  att->mutable_shape()->add_dim()->set_dim_value(3);

  // Input B: shape [28] — rank 1, triggers CVE-2026-63632
  ValueInfoProto *b = graph->add_input();
  b->set_name("B");
  auto *btt = b->mutable_type()->mutable_tensor_type();
  btt->set_elem_type(floatType);
  btt->mutable_shape()->add_dim()->set_dim_value(28);

  // Output Y
  ValueInfoProto *y = graph->add_output();
  y->set_name("Y");
  y->mutable_type()->mutable_tensor_type()->set_elem_type(floatType);

  // Gemm node
  NodeProto *node = graph->add_node();
  node->set_op_type("Gemm");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");

  return model;
}

// Build an opset-7 Gemm model where input B has rank 2 (shape [3, 4]).
// This is a valid model and should be accepted.
static ModelProto buildGemmRank2BModel() {
  ModelProto model;
  model.set_ir_version(6);

  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(7);

  GraphProto *graph = model.mutable_graph();
  graph->set_name("gemm_rank2_b");

  const auto floatType = TensorProto_DataType::TensorProto_DataType_FLOAT;

  // Input A: shape [2, 3]
  ValueInfoProto *a = graph->add_input();
  a->set_name("A");
  auto *att = a->mutable_type()->mutable_tensor_type();
  att->set_elem_type(floatType);
  att->mutable_shape()->add_dim()->set_dim_value(2);
  att->mutable_shape()->add_dim()->set_dim_value(3);

  // Input B: shape [3, 4] — valid rank-2
  ValueInfoProto *b = graph->add_input();
  b->set_name("B");
  auto *btt = b->mutable_type()->mutable_tensor_type();
  btt->set_elem_type(floatType);
  btt->mutable_shape()->add_dim()->set_dim_value(3);
  btt->mutable_shape()->add_dim()->set_dim_value(4);

  // Output Y
  ValueInfoProto *y = graph->add_output();
  y->set_name("Y");
  y->mutable_type()->mutable_tensor_type()->set_elem_type(floatType);

  // Gemm node
  NodeProto *node = graph->add_node();
  node->set_op_type("Gemm");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");

  return model;
}

// Returns 0 on pass, 1 on fail.
static int testGemmRank1BRejected() {
  mlir::MLIRContext context;
  loadDialects(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;

  onnx_mlir::ImportOptions options;
  options.invokeOnnxVersionConverter = true;

  // Serialize ModelProto to a byte buffer and import via the public API.
  ModelProto model = buildGemmRank1BModel();
  std::string buf;
  model.SerializeToString(&buf);

  int rc = onnx_mlir::ImportFrontendModelArray(
      buf.data(), (int)buf.size(), context, module, &errorMessage, options);

  if (rc == onnx_mlir::CompilerSuccess) {
    std::cerr << "FAIL: expected ImportFrontendModelArray to return "
                 "CompilerFailure for Gemm with rank-1 B input "
                 "(CVE-2026-63632)\n";
    return 1;
  }
  std::cout << "PASS: Gemm rank-1 B correctly rejected before "
               "ConvertVersion() (CVE-2026-63632)\n";
  return 0;
}

// Returns 0 on pass, 1 on fail.
static int testGemmRank2BAccepted() {
  mlir::MLIRContext context;
  loadDialects(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;

  onnx_mlir::ImportOptions options;
  options.invokeOnnxVersionConverter = true;

  ModelProto model = buildGemmRank2BModel();
  std::string buf;
  model.SerializeToString(&buf);

  int rc = onnx_mlir::ImportFrontendModelArray(
      buf.data(), (int)buf.size(), context, module, &errorMessage, options);

  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "FAIL: expected ImportFrontendModelArray to return "
                 "CompilerSuccess for Gemm with valid rank-2 B input, got: "
              << errorMessage << "\n";
    return 1;
  }
  std::cout << "PASS: Gemm rank-2 B correctly accepted by ConvertVersion()\n";
  return 0;
}

int main(int /*argc*/, char * /*argv*/[]) {
  int rc = 0;
  rc += testGemmRank1BRejected();
  rc += testGemmRank2BAccepted();
  return rc;
}
