include "mlir/IR/OpBase.td"

def Knl_Dialect : Dialect {
  let name = "knl";
  let cppNamespace = "";
}

def KnlIterate : Op<Knl_Dialect, "iterate"> {
  let summary = "iterate operation";
  let description = [{

    The "knl.iterate" operation is conceptually equivalent to a nested for loop
    in that it represents ordered interation of integer coordinates within an
    affine integer set.

  }];

  let arguments = (ins Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "IntegerSet set, ArrayRef<Value *> args">
  ];
}