module {
  func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 5>} : (tensor<3x5xf32>) -> tensor<1x3x5xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 5, 4>} : (tensor<5x4xf32>) -> tensor<1x5x4xf32>
    %2 = tosa.matmul %0, %1 : (tensor<1x3x5xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
    %3 = tosa.reshape %arg2 {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %4 = tosa.add %2, %3 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
    return %5 : tensor<3x4xf32>
  }
}


// -----
module {
  func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 6, 4>} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
    %2 = "tosa.const"() <{value = dense<1.618000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %3 = tosa.mul %2, %0 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
    %4 = tosa.matmul %3, %1 : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
    %5 = tosa.reshape %arg2 {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %6 = tosa.add %4, %5 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
    return %7 : tensor<3x4xf32>
  }
}


// -----
module {
  func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<6x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 6, 6>} : (tensor<6x6xf32>) -> tensor<1x6x6xf32>
    %2 = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %3 = tosa.reshape %arg2 {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
    %4 = tosa.mul %2, %3 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
    %5 = tosa.matmul %0, %1 : (tensor<1x3x6xf32>, tensor<1x6x6xf32>) -> tensor<1x3x6xf32>
    %6 = tosa.add %5, %4 : (tensor<1x3x6xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 3, 6>} : (tensor<1x3x6xf32>) -> tensor<3x6xf32>
    return %7 : tensor<3x6xf32>
  }
}


// -----
module {
  func.func @test_transa(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 6, 3>} : (tensor<6x3xf32>) -> tensor<1x6x3xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 6, 4>} : (tensor<6x4xf32>) -> tensor<1x6x4xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = tosa.transpose %0, %2 : (tensor<1x6x3xf32>, tensor<3xi32>) -> tensor<1x3x6xf32>
    %4 = tosa.matmul %3, %1 : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
    %5 = tosa.reshape %arg2 {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %6 = tosa.add %4, %5 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
    return %7 : tensor<3x4xf32>
  }
}


// -----
module {
  func.func @test_transb(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 3, 6>} : (tensor<3x6xf32>) -> tensor<1x3x6xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = tosa.transpose %1, %2 : (tensor<1x4x6xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
    %4 = "tosa.const"() <{value = dense<1.184000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %5 = tosa.mul %4, %0 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x3x6xf32>) -> tensor<1x3x6xf32>
    %6 = tosa.matmul %5, %3 : (tensor<1x3x6xf32>, tensor<1x6x4xf32>) -> tensor<1x3x4xf32>
    %7 = tosa.reshape %arg2 {new_shape = array<i64: 1, 3, 4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
    %8 = tosa.add %6, %7 : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %9 = tosa.reshape %8 {new_shape = array<i64: 3, 4>} : (tensor<1x3x4xf32>) -> tensor<3x4xf32>
    return %9 : tensor<3x4xf32>
  }
}


// -----
module {
  func.func @test_no_c(%arg0: tensor<1x5xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x5xf32> {
    %0 = "onnx.NoValue"() <{value}> : () -> none
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
    %2 = tosa.reshape %arg1 {new_shape = array<i64: 1, 5, 5>} : (tensor<5x5xf32>) -> tensor<1x5x5xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %4 = tosa.transpose %2, %3 : (tensor<1x5x5xf32>, tensor<3xi32>) -> tensor<1x5x5xf32>
    %5 = tosa.matmul %1, %4 : (tensor<1x1x5xf32>, tensor<1x5x5xf32>) -> tensor<1x1x5xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 5>} : (tensor<1x1x5xf32>) -> tensor<1x5xf32>
    return %6 : tensor<1x5xf32>
  }
}


// -----
module {
  func.func @test_no_c_no_trans(%arg0: tensor<1x5xf32>, %arg1: tensor<5x6xf32>) -> tensor<1x6xf32> {
    %0 = "onnx.NoValue"() <{value}> : () -> none
    %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
    %2 = tosa.reshape %arg1 {new_shape = array<i64: 1, 5, 6>} : (tensor<5x6xf32>) -> tensor<1x5x6xf32>
    %3 = "tosa.const"() <{value = dense<1.349000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %4 = tosa.mul %3, %1 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x1x5xf32>) -> tensor<1x1x5xf32>
    %5 = tosa.matmul %4, %2 : (tensor<1x1x5xf32>, tensor<1x5x6xf32>) -> tensor<1x1x6xf32>
    %6 = tosa.reshape %5 {new_shape = array<i64: 1, 6>} : (tensor<1x1x6xf32>) -> tensor<1x6xf32>
    return %6 : tensor<1x6xf32>
  }
}


// -----
module {
  func.func @test_mixed(%arg0: tensor<11x5xf32>, %arg1: tensor<3x11xf32>, %arg2: tensor<5x3xf32>) -> tensor<5x3xf32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 11, 5>} : (tensor<11x5xf32>) -> tensor<1x11x5xf32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 3, 11>} : (tensor<3x11xf32>) -> tensor<1x3x11xf32>
    %2 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %3 = tosa.transpose %0, %2 : (tensor<1x11x5xf32>, tensor<3xi32>) -> tensor<1x5x11xf32>
    %4 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %5 = tosa.transpose %1, %4 : (tensor<1x3x11xf32>, tensor<3xi32>) -> tensor<1x11x3xf32>
    %6 = "tosa.const"() <{value = dense<1.402000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %7 = tosa.mul %6, %3 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x5x11xf32>) -> tensor<1x5x11xf32>
    %8 = "tosa.const"() <{value = dense<1.998000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %9 = tosa.reshape %arg2 {new_shape = array<i64: 1, 5, 3>} : (tensor<5x3xf32>) -> tensor<1x5x3xf32>
    %10 = tosa.mul %8, %9 {shift = 0 : i32} : (tensor<1x1x1xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    %11 = tosa.matmul %7, %5 : (tensor<1x5x11xf32>, tensor<1x11x3xf32>) -> tensor<1x5x3xf32>
    %12 = tosa.add %11, %10 : (tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    %13 = tosa.reshape %12 {new_shape = array<i64: 5, 3>} : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
    return %13 : tensor<5x3xf32>
  }
}

