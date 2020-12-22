#include <iostream>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include "src/Builder/FrontendDialectTransformer.hpp"

using namespace std;
using namespace ONNX_NAMESPACE;

#define ONNX_OPSET_VERSION 11

void RegisterFunSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("SquareFn")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(ONNX_OPSET_VERSION)
      .SetDoc("This operator returns an output tensor that is twice the input "
              "tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint(
          "T", {"tensor(float)"}, "Type of the input and output values")
      .FunctionBody(FunctionBodyHelper::BuildNodes(
          {// nodes: {outputs, op, inputs, attributes}
              {{"Two"}, "Constant", {}, {{"value", ToTensor(2.0f)}}},
              {{"Y"}, "Mul", {"Two", "X"}}}));
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

void registerDialects(mlir::MLIRContext &context) {
  // mlir::DialectRegistry registry;
  // registry.insert<mlir::StandardOpsDialect>();
  // registry.insert<mlir::ONNXOpsDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::ONNXOpsDialect>();
}

void check(ModelProto &model) {
  mlir::MLIRContext context;
  registerDialects(context);
  mlir::OwningModuleRef module;

  onnx_mlir::ImportFrontendModel(model, context, module);

  module->verify();
  module->dump();
}

int main(int argc, char *argv[]) {

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
  x->mutable_type()->mutable_tensor_type()->set_elem_type(elt_type);

  auto *y = graph->add_output();
  y->set_name("y");
  y->mutable_type()->mutable_tensor_type()->set_elem_type(elt_type);

  auto *node = graph->add_node();
  node->add_input("x");
  node->add_output("y");
  node->set_op_type("SquareFn");
  node->set_name("node1");

  auto *t = graph->add_value_info();
  t->set_name("t");
  t->mutable_type()->mutable_tensor_type()->set_elem_type(elt_type);

  node = graph->add_node();
  node->add_input("x");
  node->add_output("t");
  node->set_op_type("SquareFn");

  check(model_proto);

  return 0;
}
