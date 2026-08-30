"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xi64>, sym_name = "main_graph"}> ({
  ^bb0(%arg0: tensor<1x2x3x4xf32>):
    %0 = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = i64}> : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %1 = "onnx.Transpose"(%0) <{perm = [0, 3, 1, 2]}> : (tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xi64>
    "onnx.Return"(%1) : (tensor<1x4x2x3xi64>) -> ()
  }) : () -> ()
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}) : () -> ()
