module {
  func.func @test_default_transpose(%arg0: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[3, 2, 1, 0]> : tensor<4xi64>} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
    return %0 : tensor<32x1x5x5xf32>
  }
  func.func @test_transpose(%arg0: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[3, 2, 1, 0]> : tensor<4xi64>} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
    return %0 : tensor<32x1x5x5xf32>
  }
  func.func @test_transpose_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[3, 2, 1, 0]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
}

