// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

func.func @test_range_dynamic_f32(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: test_range_dynamic
  // CHECK: [[START:%.*]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[DELTA:%.*]] = krnl.load %arg2[] : memref<f32>

  // CHECK: [[LIMIT:%.*]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[SUB:%.*]] = arith.subf [[LIMIT]], [[START]] : f32
  // CHECK: [[DIV:%.*]] = arith.divf [[SUB]], [[DELTA]] : f32
  // CHECK: [[CEIL:%.*]] = math.ceil [[DIV]] : f32
  // CHECK: [[FPTOUI:%.*]] = arith.fptoui [[CEIL]] : f32 to i64
  // CHECK: [[CAST:%.*]] = arith.index_cast [[FPTOUI]] : i64 to index
  // CHECK: [[RES:%.*]] = memref.alloc([[CAST]]) {{.*}}: memref<?xf32>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xf32>

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg3 = 0 to [[MAP_0_]]([[CAST]])){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<?xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[LOAD_ACC]], [[DELTA]] : f32
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: }

  // CHECK: return [[RES]] : memref<?xf32>
}

// -----

func.func @test_range_dynamic_f64(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<*xf64> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: test_range_dynamic_f64
  // CHECK: [[START:%.*]] = krnl.load %arg0[] : memref<f64>
  // CHECK: [[DELTA:%.*]] = krnl.load %arg2[] : memref<f64>

  // CHECK: [[LIMIT:%.*]] = krnl.load %arg1[] : memref<f64>
  // CHECK: [[SUB:%.*]] = arith.subf [[LIMIT]], [[START]] : f64
  // CHECK: [[DIV:%.*]] = arith.divf [[SUB]], [[DELTA]] : f64
  // CHECK: [[CEIL:%.*]] = math.ceil [[DIV]] : f64
  // CHECK: [[FPTOUI:%.*]] = arith.fptoui [[CEIL]] : f64 to i64
  // CHECK: [[CAST:%.*]] = arith.index_cast [[FPTOUI]] : i64 to index
  // CHECK: [[RES:%.*]] = memref.alloc([[CAST]]) {{.*}}: memref<?xf64>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xf64>

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg3 = 0 to [[MAP_0_]]([[CAST]])){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<?xf64>
  // CHECK: [[ADD:%.*]] = arith.addf [[LOAD_ACC]], [[DELTA]] : f64
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: }

  // CHECK: return [[RES]] : memref<?xf64>
}

// -----

func.func @test_range_dynamic_i16(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<i16>) -> tensor<*xi16> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<*xi16>
  return %0 : tensor<*xi16>

  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: test_range_dynamic_i16
  // CHECK: [[START:%.*]] = krnl.load %arg0[] : memref<i16>
  // CHECK: [[DELTA:%.*]] = krnl.load %arg2[] : memref<i16>

  // CHECK: [[LIMIT:%.*]] = krnl.load %arg1[] : memref<i16>
  // CHECK: [[SUB:%.*]] = arith.subi [[LIMIT]], [[START]] : i16
  // CHECK: [[CEILDIV:%.*]] = arith.ceildivsi [[SUB]], [[DELTA]] : i16
  // CHECK: [[CAST:%.*]] = arith.index_cast [[CEILDIV]] : i16 to index
  // CHECK: [[RES:%.*]] = memref.alloc([[CAST]]) {{.*}}: memref<?xi16>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi16>

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg3 = 0 to [[MAP_0_]]([[CAST]])){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<?xi16>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i16
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: }

  // CHECK: return [[RES]] : memref<?xi16>
}

// -----

func.func @test_range_dynamic_i32(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<*xi32> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: test_range_dynamic_i32
  // CHECK: [[START:%.*]] = krnl.load %arg0[] : memref<i32>
  // CHECK: [[DELTA:%.*]] = krnl.load %arg2[] : memref<i32>

  // CHECK: [[LIMIT:%.*]] = krnl.load %arg1[] : memref<i32>
  // CHECK: [[SUB:%.*]] = arith.subi [[LIMIT]], [[START]] : i32
  // CHECK: [[CEILDIV:%.*]] = arith.ceildivsi [[SUB]], [[DELTA]] : i32
  // CHECK: [[CAST:%.*]] = arith.index_cast [[CEILDIV]] : i32 to index
  // CHECK: [[RES:%.*]] = memref.alloc([[CAST]]) {{.*}}: memref<?xi32>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi32>

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg3 = 0 to [[MAP_0_]]([[CAST]])){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<?xi32>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i32
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: }

  // CHECK: return [[RES]] : memref<?xi32>
}

// -----

func.func @test_range_dynamic_i64(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<*xi64> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: test_range_dynamic_i64
  // CHECK: [[START:%.*]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[DELTA:%.*]] = krnl.load %arg2[] : memref<i64>

  // CHECK: [[LIMIT:%.*]] = krnl.load %arg1[] : memref<i64>
  // CHECK: [[SUB:%.*]] = arith.subi [[LIMIT]], [[START]] : i64
  // CHECK: [[CEILDIV:%.*]] = arith.ceildivsi [[SUB]], [[DELTA]] : i64
  // CHECK: [[CAST:%.*]] = arith.index_cast [[CEILDIV]] : i64 to index
  // CHECK: [[RES:%.*]] = memref.alloc([[CAST]]) {{.*}}: memref<?xi64>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi64>

  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg3 = 0 to [[MAP_0_]]([[CAST]])){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<?xi64>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i64
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: }

  // CHECK: return [[RES]] : memref<?xi64>
}

// -----

func.func @test_range_static_f32() -> tensor<*xf32> {
  %start = onnx.Constant dense<[2.0]> : tensor<1xf32>
  %limit = onnx.Constant dense<[10.0]> : tensor<1xf32>
  %delta = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_range_static_f32
  // CHECK: [[START_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<2.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
  // CHECK: [[LIMIT_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1.000000e+01> : tensor<1xf32>} : () -> memref<1xf32>
  // CHECK: [[DELTA_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
    
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: [[START:%.*]] = krnl.load [[START_GLOBAL]][%[[C0]]] : memref<1xf32>
  // CHECK: [[DELTA:%.*]] = krnl.load [[DELTA_GLOBAL]][%[[C0]]] : memref<1xf32>

  // CHECK: [[RES:%.*]] = memref.alloc() {{.*}}: memref<8xf32>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xf32>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg0 = 0 to 8){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<8xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[LOAD_ACC]], [[DELTA]] : f32
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xf32>
  // CHECK: }

  // CHECK: return [[RES]] : memref<8xf32>
}

// -----

func.func @test_range_static_f64() -> tensor<*xf64> {
  %start = onnx.Constant dense<[2.0]> : tensor<1xf64>
  %limit = onnx.Constant dense<[10.0]> : tensor<1xf64>
  %delta = onnx.Constant dense<[1.0]> : tensor<1xf64>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

  // CHECK-LABEL: test_range_static_f64
  // CHECK: [[START_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<2.000000e+00> : tensor<1xf64>} : () -> memref<1xf64>
  // CHECK: [[LIMIT_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1.000000e+01> : tensor<1xf64>} : () -> memref<1xf64>
  // CHECK: [[DELTA_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1.000000e+00> : tensor<1xf64>} : () -> memref<1xf64>
    
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: [[START:%.*]] = krnl.load [[START_GLOBAL]][%[[C0]]] : memref<1xf64>
  // CHECK: [[DELTA:%.*]] = krnl.load [[DELTA_GLOBAL]][%[[C0]]] : memref<1xf64>

  // CHECK: [[RES:%.*]] = memref.alloc() {{.*}}: memref<8xf64>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xf64>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg0 = 0 to 8){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<8xf64>
  // CHECK: [[ADD:%.*]] = arith.addf [[LOAD_ACC]], [[DELTA]] : f64
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xf64>
  // CHECK: }

  // CHECK: return [[RES]] : memref<8xf64>
}

// -----

func.func @test_range_static_i16() -> tensor<*xi16> {
  %start = onnx.Constant dense<[2]> : tensor<1xi16>
  %limit = onnx.Constant dense<[10]> : tensor<1xi16>
  %delta = onnx.Constant dense<[1]> : tensor<1xi16>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<*xi16>
  return %0 : tensor<*xi16>

  // CHECK-LABEL: test_range_static_i16
  // CHECK: [[START_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<2> : tensor<1xi16>} : () -> memref<1xi16>
  // CHECK: [[LIMIT_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<10> : tensor<1xi16>} : () -> memref<1xi16>
  // CHECK: [[DELTA_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1> : tensor<1xi16>} : () -> memref<1xi16>
    
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: [[START:%.*]] = krnl.load [[START_GLOBAL]][%[[C0]]] : memref<1xi16>
  // CHECK: [[DELTA:%.*]] = krnl.load [[DELTA_GLOBAL]][%[[C0]]] : memref<1xi16>

  // CHECK: [[RES:%.*]] = memref.alloc() {{.*}}: memref<8xi16>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi16>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg0 = 0 to 8){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<8xi16>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i16
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi16>
  // CHECK: }

  // CHECK: return [[RES]] : memref<8xi16>
}

// -----

func.func @test_range_static_i32() -> tensor<*xi32> {
  %start = onnx.Constant dense<[2]> : tensor<1xi32>
  %limit = onnx.Constant dense<[10]> : tensor<1xi32>
  %delta = onnx.Constant dense<[1]> : tensor<1xi32>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

  // CHECK-LABEL: test_range_static_i32
  // CHECK: [[START_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<2> : tensor<1xi32>} : () -> memref<1xi32>
  // CHECK: [[LIMIT_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<10> : tensor<1xi32>} : () -> memref<1xi32>
  // CHECK: [[DELTA_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1> : tensor<1xi32>} : () -> memref<1xi32>
    
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: [[START:%.*]] = krnl.load [[START_GLOBAL]][%[[C0]]] : memref<1xi32>
  // CHECK: [[DELTA:%.*]] = krnl.load [[DELTA_GLOBAL]][%[[C0]]] : memref<1xi32>

  // CHECK: [[RES:%.*]] = memref.alloc() {{.*}}: memref<8xi32>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi32>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg0 = 0 to 8){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<8xi32>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i32
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi32>
  // CHECK: }

  // CHECK: return [[RES]] : memref<8xi32>
}

// -----

func.func @test_range_static_i64() -> tensor<*xi64> {
  %start = onnx.Constant dense<[2]> : tensor<1xi64>
  %limit = onnx.Constant dense<[10]> : tensor<1xi64>
  %delta = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: test_range_static_i64
  // CHECK: [[START_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<2> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK: [[LIMIT_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<10> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK: [[DELTA_GLOBAL:%.*]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
    
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: [[START:%.*]] = krnl.load [[START_GLOBAL]][%[[C0]]] : memref<1xi64>
  // CHECK: [[DELTA:%.*]] = krnl.load [[DELTA_GLOBAL]][%[[C0]]] : memref<1xi64>

  // CHECK: [[RES:%.*]] = memref.alloc() {{.*}}: memref<8xi64>
  // CHECK: [[ACC:%.*]] = memref.alloc() {{.*}}: memref<1xi64>
  // CHECK: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK: krnl.store [[START]], [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: [[LOOP:%.*]] = krnl.define_loops 1

  // CHECK: krnl.iterate([[LOOP]]) with ([[LOOP]] -> %arg0 = 0 to 8){
  // CHECK: [[IV:%.*]] = krnl.get_induction_var_value([[LOOP]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD_ACC:%.*]] = krnl.load [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: krnl.store [[LOAD_ACC]], [[RES]][[[IV]]] : memref<8xi64>
  // CHECK: [[ADD:%.*]] = arith.addi [[LOAD_ACC]], [[DELTA]] : i64
  // CHECK: krnl.store [[ADD]], [[ACC]][%[[C0_0]]] : memref<1xi64>
  // CHECK: }

  // CHECK: return [[RES]] : memref<8xi64>
}
