#ifdef FOLD_INTERFACE
#else
#define FOLD_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def FoldOpInterface : OpInterface<"FoldInterface"> {
  let description = [{
    Interface to access a registered method to fold an operation.
    Depends on the inferShapes() of the operation being run first.
  }];

  let methods = [
    InterfaceMethod<"Best effort folding of the Op.",
                    "void", "tryFold">
  ];
}

#endif // FOLD_INTERFACE
