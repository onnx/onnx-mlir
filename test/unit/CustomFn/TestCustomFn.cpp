/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/SuppressWarnings.h"

SUPPRESS_WARNINGS_PUSH
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
SUPPRESS_WARNINGS_POP

using namespace std;
using namespace ONNX_NAMESPACE;

#define ONNX_OPSET_VERSION 11

void Register(ONNX_NAMESPACE::OpSchema &schema) {
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

void RegisterFunSchema() {
  static bool registered = false;
  if (registered)
    return;
  ONNX_NAMESPACE::OpSchema twiceSchema;
  twiceSchema.SetName("TwiceFn")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(ONNX_OPSET_VERSION)
      .SetDoc("This operator returns an output tensor that is twice the input "
              "tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint(
          "T", {"tensor(float)"}, "Type of the input and output values")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
      .FunctionBody(FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
              {{"Two"}, "Constant", {}, {{"value", ToTensor(2.0f)}}},
              {{"Y"}, "Mul", {"Two", "X"}}}));
  Register(twiceSchema);

  ONNX_NAMESPACE::OpSchema ttSchema;
  ttSchema.SetName("TwiceTwiceFn")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(ONNX_OPSET_VERSION)
      .SetDoc(
          "This operator returns an output tensor that is four times the input "
          "tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint(
          "T", {"tensor(float)"}, "Type of the input and output values")
      .FunctionBody(FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
              {{"T"}, "TwiceFn", {"X"}}, {{"Y"}, "TwiceFn", {"T"}}}));
  Register(ttSchema);
  registered = true;
}

void loadDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::ONNXDialect>();
}

int check(ModelProto &model) {
  mlir::MLIRContext context;
  loadDialects(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;

  onnx_mlir::ImportOptions options;
  options.useOnnxModelTypes = true;
  onnx_mlir::ImportFrontendModel(model, context, module, options);

  mlir::PassManager pm(
      module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
  pm.addNestedPass<mlir::func::FuncOp>(onnx_mlir::createShapeInferencePass());
  (void)mlir::applyPassManagerCLOptions(pm);
  if (mlir::failed(pm.run(*module))) {
    module->dump();
    std::cerr << "Error applying shape inference!\n";
    return 1;
  }

  if (mlir::failed(module->verifyInvariants())) {
    module->dump();
    std::cerr << "Error verifying module!\n";
    return 1;
  }
  module->dump();
  std::cerr << "\n\n";
  return 0;
}

int testTranslation(const char *fnname) {
  RegisterFunSchema();

  ModelProto model_proto;
  model_proto.set_ir_version(7);
  auto *opset_version = model_proto.add_opset_import();
  opset_version->set_domain(ONNX_DOMAIN);
  opset_version->set_version(ONNX_OPSET_VERSION);

  auto *graph = model_proto.mutable_graph();

  auto elt_type = TensorProto_DataType::TensorProto_DataType_FLOAT;

  auto *x = graph->add_input();
  x->set_name("x");
  auto *x_type = x->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(elt_type);
  auto *x_shape = x_type->mutable_shape();
  x_shape->add_dim()->set_dim_value(10);

  auto *y = graph->add_output();
  y->set_name("y");
  auto *y_type = y->mutable_type()->mutable_tensor_type();
  y_type->set_elem_type(elt_type);
  auto *y_shape = y_type->mutable_shape();
  y_shape->add_dim()->set_dim_value(10);

  auto *node = graph->add_node();
  node->add_input("x");
  node->add_output("y");
  node->set_op_type(fnname);
  node->set_name("node1");

  return check(model_proto);
}

int testCustomFunTranslation() { return testTranslation("TwiceFn"); }

int testNestedFunTranslation() { return testTranslation("TwiceTwiceFn"); }

int testUseOfOnnxModelTypes() {
  RegisterFunSchema();

  ModelProto model_proto;
  model_proto.set_ir_version(7);
  auto *opset_version = model_proto.add_opset_import();
  opset_version->set_domain(ONNX_DOMAIN);
  opset_version->set_version(ONNX_OPSET_VERSION);

  auto *graph = model_proto.mutable_graph();

  auto float_type = TensorProto_DataType::TensorProto_DataType_FLOAT;
  auto int_type = TensorProto_DataType::TensorProto_DataType_INT32;

  auto *x = graph->add_input();
  x->set_name("x");
  auto *x_type = x->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(float_type);
  auto *x_shape = x_type->mutable_shape();
  x_shape->add_dim()->set_dim_value(10);

  auto *t = graph->add_value_info();
  t->set_name("t");
  auto *t_type = t->mutable_type()->mutable_tensor_type();
  t_type->set_elem_type(int_type);
  auto *t_shape = t_type->mutable_shape();
  t_shape->add_dim()->set_dim_value(10);

  auto *y = graph->add_output();
  y->set_name("y");
  auto *y_type = y->mutable_type()->mutable_tensor_type();
  y_type->set_elem_type(float_type);
  auto *y_shape = y_type->mutable_shape();
  y_shape->add_dim()->set_dim_value(10);

  auto *node = graph->add_node();
  node->add_input("x");
  node->add_output("t");
  node->set_op_type("CustomFn1");

  node = graph->add_node();
  node->add_input("t");
  node->add_output("y");
  node->set_op_type("CustomFn2");

  return check(model_proto);
}

void RegisterOptParamFunSchema() {
  static bool registered = false;
  if (registered)
    return;
  ONNX_NAMESPACE::OpSchema schema;
  // A sample function with an optional first parameter.
  schema.SetName("TestFun1")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(ONNX_OPSET_VERSION)
      .SetDoc("This operator returns the second input.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Optional)
      .Input(1, "Y", "Input tensor", "T", OpSchema::Single)
      .Input(2, "W", "Input tensor", "T", OpSchema::Optional)
      .Output(0, "Z", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint(
          "T", {"tensor(float)"}, "Type of the input and output values")
      .FunctionBody(FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
              {{"Z"}, "Identity", {"Y"}}}));
  Register(schema);
  registered = true;
}

int testOptionalParameter() {
  RegisterOptParamFunSchema();

  ModelProto model_proto;
  model_proto.set_ir_version(7);
  auto *opset_version = model_proto.add_opset_import();
  opset_version->set_domain(ONNX_DOMAIN);
  opset_version->set_version(ONNX_OPSET_VERSION);

  auto *graph = model_proto.mutable_graph();

  auto float_type = TensorProto_DataType::TensorProto_DataType_FLOAT;

  auto *x = graph->add_input();
  x->set_name("x");
  auto *x_type = x->mutable_type()->mutable_tensor_type();
  x_type->set_elem_type(float_type);
  auto *x_shape = x_type->mutable_shape();
  x_shape->add_dim()->set_dim_value(10);

  auto *y = graph->add_output();
  y->set_name("y");
  auto *y_type = y->mutable_type()->mutable_tensor_type();
  y_type->set_elem_type(float_type);
  auto *y_shape = y_type->mutable_shape();
  y_shape->add_dim()->set_dim_value(10);

  auto *z = graph->add_output();
  z->set_name("z");
  auto *z_type = z->mutable_type()->mutable_tensor_type();
  z_type->set_elem_type(float_type);
  auto *z_shape = z_type->mutable_shape();
  z_shape->add_dim()->set_dim_value(10);

  // An invocation with optional parameter absent:
  auto *node = graph->add_node();
  node->add_input("");
  node->add_input("x");
  node->add_output("y");
  node->set_op_type("TestFun1");

  // An invocation with optional parameter present:
  node = graph->add_node();
  node->add_input("y");
  node->add_input("x");
  node->add_output("z");
  node->set_op_type("TestFun1");

  return check(model_proto);
}

int main(int argc, char *argv[]) {
  int status = 0;

  status |= testCustomFunTranslation();
  status |= testNestedFunTranslation();
  status |= testUseOfOnnxModelTypes();
  status |= testOptionalParameter();

  return status;
}
