module {
  func.func @test_nonmaxsuppression_center_point_box_format(%arg0: memref<1x6x4xf32>, %arg1: memref<1x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    %alloca_3 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_3[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %14:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_2[] : memref<index>
      %15 = krnl.define_loops 1
      krnl.iterate(%15) with (%15 -> %arg7 = %c0 to %c6){
        %19 = krnl.get_induction_var_value(%15) : (!krnl.loop) -> index
        %20 = krnl.load %arg1[%14#0, %14#1, %19] : memref<1x1x6xf32>
        %21 = arith.cmpf ogt, %20, %1 : f32
        %22 = krnl.load %alloca_2[] : memref<index>
        %23 = arith.addi %22, %c1 : index
        %24 = arith.select %21, %23, %22 : index
        krnl.store %24, %alloca_2[] : memref<index>
      }
      %16 = krnl.load %alloca_2[] : memref<index>
      %17 = krnl.load %alloca_3[] : memref<index>
      %18 = arith.maxsi %16, %17 : index
      krnl.store %18, %alloca_3[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_3[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %14:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %14#2, %alloc[%14#0, %14#1, %14#2] : memref<1x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = %c0 to %c1, %11#1 -> %arg6 = %c0 to %c1){
      %14:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %15 = krnl.define_loops 1
      krnl.iterate(%15) with (%15 -> %arg7 = %c0 to %c6){
        %16 = krnl.get_induction_var_value(%15) : (!krnl.loop) -> index
        %17 = krnl.load %alloc[%14#0, %14#1, %16] : memref<1x1x6xindex>
        %18 = krnl.load %arg1[%14#0, %14#1, %17] : memref<1x1x6xf32>
        %19 = arith.cmpf ogt, %18, %1 : f32
        %20 = krnl.load %alloca_6[] : memref<index>
        %21 = arith.cmpi slt, %20, %9 : index
        %22 = krnl.load %alloc_8[%17] : memref<6xi1>
        %23 = arith.cmpi eq, %22, %false : i1
        %24 = arith.andi %19, %21 : i1
        %25 = arith.andi %24, %23 : i1
        scf.if %25 {
          %26 = krnl.load %arg0[%14#0, %17, %c0] : memref<1x6x4xf32>
          %27 = krnl.load %arg0[%14#0, %17, %c1] : memref<1x6x4xf32>
          %28 = krnl.load %arg0[%14#0, %17, %c2] : memref<1x6x4xf32>
          %29 = krnl.load %arg0[%14#0, %17, %c3] : memref<1x6x4xf32>
          %30 = krnl.load %alloca_5[] : memref<index>
          krnl.store %14#0, %alloc_4[%30, %c0] : memref<?x3xindex>
          krnl.store %14#1, %alloc_4[%30, %c1] : memref<?x3xindex>
          krnl.store %17, %alloc_4[%30, %c2] : memref<?x3xindex>
          %31 = arith.addi %20, %c1 : index
          krnl.store %31, %alloca_6[] : memref<index>
          %32 = arith.addi %30, %c1 : index
          krnl.store %32, %alloca_5[] : memref<index>
          %33 = krnl.define_loops 1
          krnl.iterate(%33) with (%33 -> %arg8 = %c0 to %c6){
            %34 = krnl.get_induction_var_value(%33) : (!krnl.loop) -> index
            %35 = krnl.load %alloc_8[%34] : memref<6xi1>
            %36 = arith.cmpi eq, %35, %false : i1
            scf.if %36 {
              %37 = krnl.load %arg0[%14#0, %34, %c0] : memref<1x6x4xf32>
              %38 = krnl.load %arg0[%14#0, %34, %c1] : memref<1x6x4xf32>
              %39 = krnl.load %arg0[%14#0, %34, %c2] : memref<1x6x4xf32>
              %40 = krnl.load %arg0[%14#0, %34, %c3] : memref<1x6x4xf32>
              %41 = arith.divf %28, %cst_1 : f32
              %42 = arith.subf %26, %41 : f32
              %43 = arith.divf %28, %cst_1 : f32
              %44 = arith.addf %26, %43 : f32
              %45 = arith.divf %29, %cst_1 : f32
              %46 = arith.subf %27, %45 : f32
              %47 = arith.divf %29, %cst_1 : f32
              %48 = arith.addf %27, %47 : f32
              %49 = arith.divf %40, %cst_1 : f32
              %50 = arith.subf %38, %49 : f32
              %51 = arith.divf %40, %cst_1 : f32
              %52 = arith.addf %38, %51 : f32
              %53 = arith.divf %39, %cst_1 : f32
              %54 = arith.subf %37, %53 : f32
              %55 = arith.divf %39, %cst_1 : f32
              %56 = arith.addf %37, %55 : f32
              %57 = arith.mulf %29, %28 : f32
              %58 = arith.mulf %40, %39 : f32
              %59 = arith.maxf %42, %54 : f32
              %60 = arith.maxf %46, %50 : f32
              %61 = arith.minf %44, %56 : f32
              %62 = arith.minf %48, %52 : f32
              %63 = arith.subf %61, %59 : f32
              %64 = arith.subf %62, %60 : f32
              %65 = arith.maxf %64, %cst_0 : f32
              %66 = arith.maxf %63, %cst_0 : f32
              %67 = arith.mulf %66, %65 : f32
              %68 = arith.addf %57, %58 : f32
              %69 = arith.subf %68, %67 : f32
              %70 = arith.addf %69, %cst : f32
              %71 = arith.divf %67, %70 : f32
              %72 = arith.cmpf oge, %71, %2 : f32
              scf.if %72 {
                krnl.store %true, %alloc_8[%34] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %12 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%12) {alignment = 16 : i64} : memref<?x3xi64>
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg5 = %c0 to %12, %13#1 -> %arg6 = %c0 to %c3){
      %14:2 = krnl.get_induction_var_value(%13#0, %13#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %15 = krnl.load %alloc_4[%14#0, %14#1] : memref<?x3xindex>
      %16 = arith.index_cast %15 : index to i64
      krnl.store %16, %alloc_7[%14#0, %14#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_flipped_coordinates(%arg0: memref<1x6x4xf32>, %arg1: memref<1x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x6xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 6){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x6x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x6x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x6xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x6xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<6xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x6x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x6x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x6x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x6x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c6){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<6xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x6x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x6x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x6x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x6x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_identical_boxes(%arg0: memref<1x10x4xf32>, %arg1: memref<1x1x10xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c10 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c10){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x10xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x10xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 10){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x10xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x10xindex>, memref<1x1x10xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x10x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 10){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x10x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x10x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x10x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x10x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x10x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x10x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x10x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x10x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<10xi1>
      krnl.memset %alloc_8, %false : memref<10xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c10){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x10xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x10xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<10xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x10x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x10x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x10x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x10x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c10){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<10xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x10x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x10x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x10x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x10x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<10xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_limit_output_size(%arg0: memref<1x6x4xf32>, %arg1: memref<1x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x6xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 6){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x6x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x6x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x6xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x6xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<6xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x6x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x6x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x6x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x6x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c6){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<6xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x6x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x6x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x6x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x6x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_single_box(%arg0: memref<1x1x4xf32>, %arg1: memref<1x1x1xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c1 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c1){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x1xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x1xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 1){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x1xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x1xindex>, memref<1x1x1xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x1x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 1){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x1x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x1x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x1x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x1x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x1x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x1x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x1x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x1x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<1xi1>
      krnl.memset %alloc_8, %false : memref<1xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c1){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x1xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x1xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<1xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x1x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x1x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x1x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x1x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c1){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<1xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x1x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x1x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x1x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x1x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<1xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_suppress_by_IOU(%arg0: memref<1x6x4xf32>, %arg1: memref<1x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x6xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 6){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x6x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x6x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x6xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x6xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<6xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x6x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x6x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x6x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x6x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c6){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<6xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x6x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x6x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x6x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x6x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
module {
  func.func @test_nonmaxsuppression_suppress_by_IOU_and_scores(%arg0: memref<1x6x4xf32>, %arg1: memref<1x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %20 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %21 = krnl.load %arg1[%15#0, %15#1, %20] : memref<1x1x6xf32>
        %22 = arith.cmpf ogt, %21, %1 : f32
        %23 = krnl.load %alloca_1[] : memref<index>
        %24 = arith.addi %23, %c1 : index
        %25 = arith.select %22, %24, %23 : index
        krnl.store %25, %alloca_1[] : memref<index>
      }
      %17 = krnl.load %alloca_1[] : memref<index>
      %18 = krnl.load %alloca_2[] : memref<index>
      %19 = arith.maxsi %17, %18 : index
      krnl.store %19, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %15:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %15#2, %alloc[%15#0, %15#1, %15#2] : memref<1x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x1x6xindex>, memref<1x1x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 6){
      %15:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %arg0[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      %17 = krnl.load %arg0[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      %18 = krnl.load %arg0[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      %19 = krnl.load %arg0[%15#0, %15#1, %c3] : memref<1x6x4xf32>
      %20 = arith.cmpf ogt, %17, %19 : f32
      %21 = arith.select %20, %19, %17 : f32
      %22 = arith.select %20, %17, %19 : f32
      %23 = arith.cmpf ogt, %16, %18 : f32
      %24 = arith.select %23, %18, %16 : f32
      %25 = arith.select %23, %16, %18 : f32
      krnl.store %24, %alloc_3[%15#0, %15#1, %c0] : memref<1x6x4xf32>
      krnl.store %21, %alloc_3[%15#0, %15#1, %c1] : memref<1x6x4xf32>
      krnl.store %25, %alloc_3[%15#0, %15#1, %c2] : memref<1x6x4xf32>
      krnl.store %22, %alloc_3[%15#0, %15#1, %c3] : memref<1x6x4xf32>
    }
    %alloc_4 = memref.alloc(%9) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg5 = %c0 to %c1, %12#1 -> %arg6 = %c0 to %c1){
      %15:2 = krnl.get_induction_var_value(%12#0, %12#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %16 = krnl.define_loops 1
      krnl.iterate(%16) with (%16 -> %arg7 = %c0 to %c6){
        %17 = krnl.get_induction_var_value(%16) : (!krnl.loop) -> index
        %18 = krnl.load %alloc[%15#0, %15#1, %17] : memref<1x1x6xindex>
        %19 = krnl.load %arg1[%15#0, %15#1, %18] : memref<1x1x6xf32>
        %20 = arith.cmpf ogt, %19, %1 : f32
        %21 = krnl.load %alloca_6[] : memref<index>
        %22 = arith.cmpi slt, %21, %9 : index
        %23 = krnl.load %alloc_8[%18] : memref<6xi1>
        %24 = arith.cmpi eq, %23, %false : i1
        %25 = arith.andi %20, %22 : i1
        %26 = arith.andi %25, %24 : i1
        scf.if %26 {
          %27 = krnl.load %alloc_3[%15#0, %18, %c0] : memref<1x6x4xf32>
          %28 = krnl.load %alloc_3[%15#0, %18, %c1] : memref<1x6x4xf32>
          %29 = krnl.load %alloc_3[%15#0, %18, %c2] : memref<1x6x4xf32>
          %30 = krnl.load %alloc_3[%15#0, %18, %c3] : memref<1x6x4xf32>
          %31 = krnl.load %alloca_5[] : memref<index>
          krnl.store %15#0, %alloc_4[%31, %c0] : memref<?x3xindex>
          krnl.store %15#1, %alloc_4[%31, %c1] : memref<?x3xindex>
          krnl.store %18, %alloc_4[%31, %c2] : memref<?x3xindex>
          %32 = arith.addi %21, %c1 : index
          krnl.store %32, %alloca_6[] : memref<index>
          %33 = arith.addi %31, %c1 : index
          krnl.store %33, %alloca_5[] : memref<index>
          %34 = krnl.define_loops 1
          krnl.iterate(%34) with (%34 -> %arg8 = %c0 to %c6){
            %35 = krnl.get_induction_var_value(%34) : (!krnl.loop) -> index
            %36 = krnl.load %alloc_8[%35] : memref<6xi1>
            %37 = arith.cmpi eq, %36, %false : i1
            scf.if %37 {
              %38 = krnl.load %alloc_3[%15#0, %35, %c0] : memref<1x6x4xf32>
              %39 = krnl.load %alloc_3[%15#0, %35, %c1] : memref<1x6x4xf32>
              %40 = krnl.load %alloc_3[%15#0, %35, %c2] : memref<1x6x4xf32>
              %41 = krnl.load %alloc_3[%15#0, %35, %c3] : memref<1x6x4xf32>
              %42 = arith.subf %30, %28 : f32
              %43 = arith.subf %29, %27 : f32
              %44 = arith.mulf %43, %42 : f32
              %45 = arith.subf %41, %39 : f32
              %46 = arith.subf %40, %38 : f32
              %47 = arith.mulf %46, %45 : f32
              %48 = arith.maxf %28, %39 : f32
              %49 = arith.maxf %27, %38 : f32
              %50 = arith.minf %30, %41 : f32
              %51 = arith.minf %29, %40 : f32
              %52 = arith.subf %50, %48 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.maxf %53, %cst_0 : f32
              %55 = arith.maxf %52, %cst_0 : f32
              %56 = arith.mulf %55, %54 : f32
              %57 = arith.addf %44, %47 : f32
              %58 = arith.subf %57, %56 : f32
              %59 = arith.addf %58, %cst : f32
              %60 = arith.divf %56, %59 : f32
              %61 = arith.cmpf oge, %60, %2 : f32
              scf.if %61 {
                krnl.store %true, %alloc_8[%35] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %13 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%13) {alignment = 16 : i64} : memref<?x3xi64>
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg5 = %c0 to %13, %14#1 -> %arg6 = %c0 to %c3){
      %15:2 = krnl.get_induction_var_value(%14#0, %14#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %16 = krnl.load %alloc_4[%15#0, %15#1] : memref<?x3xindex>
      %17 = arith.index_cast %16 : index to i64
      krnl.store %17, %alloc_7[%15#0, %15#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
#map = affine_map<()[s0] -> (s0 * 2)>
module {
  func.func @test_nonmaxsuppression_two_batches(%arg0: memref<2x6x4xf32>, %arg1: memref<2x1x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c2, %5#1 -> %arg6 = %c0 to %c1){
      %16:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %c6){
        %21 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %22 = krnl.load %arg1[%16#0, %16#1, %21] : memref<2x1x6xf32>
        %23 = arith.cmpf ogt, %22, %1 : f32
        %24 = krnl.load %alloca_1[] : memref<index>
        %25 = arith.addi %24, %c1 : index
        %26 = arith.select %23, %25, %24 : index
        krnl.store %26, %alloca_1[] : memref<index>
      }
      %18 = krnl.load %alloca_1[] : memref<index>
      %19 = krnl.load %alloca_2[] : memref<index>
      %20 = arith.maxsi %18, %19 : index
      krnl.store %20, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x1x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 2, %10#1 -> %arg6 = 0 to 1, %10#2 -> %arg7 = 0 to 6){
      %16:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %16#2, %alloc[%16#0, %16#1, %16#2] : memref<2x1x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<2x1x6xindex>, memref<2x1x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<2x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 2, %11#1 -> %arg6 = 0 to 6){
      %16:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %17 = krnl.load %arg0[%16#0, %16#1, %c0] : memref<2x6x4xf32>
      %18 = krnl.load %arg0[%16#0, %16#1, %c1] : memref<2x6x4xf32>
      %19 = krnl.load %arg0[%16#0, %16#1, %c2] : memref<2x6x4xf32>
      %20 = krnl.load %arg0[%16#0, %16#1, %c3] : memref<2x6x4xf32>
      %21 = arith.cmpf ogt, %18, %20 : f32
      %22 = arith.select %21, %20, %18 : f32
      %23 = arith.select %21, %18, %20 : f32
      %24 = arith.cmpf ogt, %17, %19 : f32
      %25 = arith.select %24, %19, %17 : f32
      %26 = arith.select %24, %17, %19 : f32
      krnl.store %25, %alloc_3[%16#0, %16#1, %c0] : memref<2x6x4xf32>
      krnl.store %22, %alloc_3[%16#0, %16#1, %c1] : memref<2x6x4xf32>
      krnl.store %26, %alloc_3[%16#0, %16#1, %c2] : memref<2x6x4xf32>
      krnl.store %23, %alloc_3[%16#0, %16#1, %c3] : memref<2x6x4xf32>
    }
    %12 = affine.apply #map()[%9]
    %alloc_4 = memref.alloc(%12) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg5 = %c0 to %c2, %13#1 -> %arg6 = %c0 to %c1){
      %16:2 = krnl.get_induction_var_value(%13#0, %13#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %c6){
        %18 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %19 = krnl.load %alloc[%16#0, %16#1, %18] : memref<2x1x6xindex>
        %20 = krnl.load %arg1[%16#0, %16#1, %19] : memref<2x1x6xf32>
        %21 = arith.cmpf ogt, %20, %1 : f32
        %22 = krnl.load %alloca_6[] : memref<index>
        %23 = arith.cmpi slt, %22, %9 : index
        %24 = krnl.load %alloc_8[%19] : memref<6xi1>
        %25 = arith.cmpi eq, %24, %false : i1
        %26 = arith.andi %21, %23 : i1
        %27 = arith.andi %26, %25 : i1
        scf.if %27 {
          %28 = krnl.load %alloc_3[%16#0, %19, %c0] : memref<2x6x4xf32>
          %29 = krnl.load %alloc_3[%16#0, %19, %c1] : memref<2x6x4xf32>
          %30 = krnl.load %alloc_3[%16#0, %19, %c2] : memref<2x6x4xf32>
          %31 = krnl.load %alloc_3[%16#0, %19, %c3] : memref<2x6x4xf32>
          %32 = krnl.load %alloca_5[] : memref<index>
          krnl.store %16#0, %alloc_4[%32, %c0] : memref<?x3xindex>
          krnl.store %16#1, %alloc_4[%32, %c1] : memref<?x3xindex>
          krnl.store %19, %alloc_4[%32, %c2] : memref<?x3xindex>
          %33 = arith.addi %22, %c1 : index
          krnl.store %33, %alloca_6[] : memref<index>
          %34 = arith.addi %32, %c1 : index
          krnl.store %34, %alloca_5[] : memref<index>
          %35 = krnl.define_loops 1
          krnl.iterate(%35) with (%35 -> %arg8 = %c0 to %c6){
            %36 = krnl.get_induction_var_value(%35) : (!krnl.loop) -> index
            %37 = krnl.load %alloc_8[%36] : memref<6xi1>
            %38 = arith.cmpi eq, %37, %false : i1
            scf.if %38 {
              %39 = krnl.load %alloc_3[%16#0, %36, %c0] : memref<2x6x4xf32>
              %40 = krnl.load %alloc_3[%16#0, %36, %c1] : memref<2x6x4xf32>
              %41 = krnl.load %alloc_3[%16#0, %36, %c2] : memref<2x6x4xf32>
              %42 = krnl.load %alloc_3[%16#0, %36, %c3] : memref<2x6x4xf32>
              %43 = arith.subf %31, %29 : f32
              %44 = arith.subf %30, %28 : f32
              %45 = arith.mulf %44, %43 : f32
              %46 = arith.subf %42, %40 : f32
              %47 = arith.subf %41, %39 : f32
              %48 = arith.mulf %47, %46 : f32
              %49 = arith.maxf %29, %40 : f32
              %50 = arith.maxf %28, %39 : f32
              %51 = arith.minf %31, %42 : f32
              %52 = arith.minf %30, %41 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.subf %52, %50 : f32
              %55 = arith.maxf %54, %cst_0 : f32
              %56 = arith.maxf %53, %cst_0 : f32
              %57 = arith.mulf %56, %55 : f32
              %58 = arith.addf %45, %48 : f32
              %59 = arith.subf %58, %57 : f32
              %60 = arith.addf %59, %cst : f32
              %61 = arith.divf %57, %60 : f32
              %62 = arith.cmpf oge, %61, %2 : f32
              scf.if %62 {
                krnl.store %true, %alloc_8[%36] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %14 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%14) {alignment = 16 : i64} : memref<?x3xi64>
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg5 = %c0 to %14, %15#1 -> %arg6 = %c0 to %c3){
      %16:2 = krnl.get_induction_var_value(%15#0, %15#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %17 = krnl.load %alloc_4[%16#0, %16#1] : memref<?x3xindex>
      %18 = arith.index_cast %17 : index to i64
      krnl.store %18, %alloc_7[%16#0, %16#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
#map = affine_map<()[s0] -> (s0 * 2)>
module {
  func.func @test_nonmaxsuppression_two_classes(%arg0: memref<1x6x4xf32>, %arg1: memref<1x2x6xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> attributes {input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], output_names = ["selected_indices"]} {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %c6 : index
    krnl.store %4, %alloca[] : memref<index>
    %alloca_1 = memref.alloca() : memref<index>
    %alloca_2 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_2[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %c1, %5#1 -> %arg6 = %c0 to %c2){
      %16:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_1[] : memref<index>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %c6){
        %21 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %22 = krnl.load %arg1[%16#0, %16#1, %21] : memref<1x2x6xf32>
        %23 = arith.cmpf ogt, %22, %1 : f32
        %24 = krnl.load %alloca_1[] : memref<index>
        %25 = arith.addi %24, %c1 : index
        %26 = arith.select %23, %25, %24 : index
        krnl.store %26, %alloca_1[] : memref<index>
      }
      %18 = krnl.load %alloca_1[] : memref<index>
      %19 = krnl.load %alloca_2[] : memref<index>
      %20 = arith.maxsi %18, %19 : index
      krnl.store %20, %alloca_2[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_2[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x2x6xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to 1, %10#1 -> %arg6 = 0 to 2, %10#2 -> %arg7 = 0 to 6){
      %16:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %16#2, %alloc[%16#0, %16#1, %16#2] : memref<1x2x6xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<1x2x6xindex>, memref<1x2x6xf32>, i64, i64) -> ()
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x6x4xf32>
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg5 = 0 to 1, %11#1 -> %arg6 = 0 to 6){
      %16:2 = krnl.get_induction_var_value(%11#0, %11#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %17 = krnl.load %arg0[%16#0, %16#1, %c0] : memref<1x6x4xf32>
      %18 = krnl.load %arg0[%16#0, %16#1, %c1] : memref<1x6x4xf32>
      %19 = krnl.load %arg0[%16#0, %16#1, %c2] : memref<1x6x4xf32>
      %20 = krnl.load %arg0[%16#0, %16#1, %c3] : memref<1x6x4xf32>
      %21 = arith.cmpf ogt, %18, %20 : f32
      %22 = arith.select %21, %20, %18 : f32
      %23 = arith.select %21, %18, %20 : f32
      %24 = arith.cmpf ogt, %17, %19 : f32
      %25 = arith.select %24, %19, %17 : f32
      %26 = arith.select %24, %17, %19 : f32
      krnl.store %25, %alloc_3[%16#0, %16#1, %c0] : memref<1x6x4xf32>
      krnl.store %22, %alloc_3[%16#0, %16#1, %c1] : memref<1x6x4xf32>
      krnl.store %26, %alloc_3[%16#0, %16#1, %c2] : memref<1x6x4xf32>
      krnl.store %23, %alloc_3[%16#0, %16#1, %c3] : memref<1x6x4xf32>
    }
    %12 = affine.apply #map()[%9]
    %alloc_4 = memref.alloc(%12) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_4, %c-1 : memref<?x3xindex>
    %alloca_5 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_5[] : memref<index>
    %alloca_6 = memref.alloca() : memref<index>
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg5 = %c0 to %c1, %13#1 -> %arg6 = %c0 to %c2){
      %16:2 = krnl.get_induction_var_value(%13#0, %13#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_6[] : memref<index>
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<6xi1>
      krnl.memset %alloc_8, %false : memref<6xi1>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %c6){
        %18 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %19 = krnl.load %alloc[%16#0, %16#1, %18] : memref<1x2x6xindex>
        %20 = krnl.load %arg1[%16#0, %16#1, %19] : memref<1x2x6xf32>
        %21 = arith.cmpf ogt, %20, %1 : f32
        %22 = krnl.load %alloca_6[] : memref<index>
        %23 = arith.cmpi slt, %22, %9 : index
        %24 = krnl.load %alloc_8[%19] : memref<6xi1>
        %25 = arith.cmpi eq, %24, %false : i1
        %26 = arith.andi %21, %23 : i1
        %27 = arith.andi %26, %25 : i1
        scf.if %27 {
          %28 = krnl.load %alloc_3[%16#0, %19, %c0] : memref<1x6x4xf32>
          %29 = krnl.load %alloc_3[%16#0, %19, %c1] : memref<1x6x4xf32>
          %30 = krnl.load %alloc_3[%16#0, %19, %c2] : memref<1x6x4xf32>
          %31 = krnl.load %alloc_3[%16#0, %19, %c3] : memref<1x6x4xf32>
          %32 = krnl.load %alloca_5[] : memref<index>
          krnl.store %16#0, %alloc_4[%32, %c0] : memref<?x3xindex>
          krnl.store %16#1, %alloc_4[%32, %c1] : memref<?x3xindex>
          krnl.store %19, %alloc_4[%32, %c2] : memref<?x3xindex>
          %33 = arith.addi %22, %c1 : index
          krnl.store %33, %alloca_6[] : memref<index>
          %34 = arith.addi %32, %c1 : index
          krnl.store %34, %alloca_5[] : memref<index>
          %35 = krnl.define_loops 1
          krnl.iterate(%35) with (%35 -> %arg8 = %c0 to %c6){
            %36 = krnl.get_induction_var_value(%35) : (!krnl.loop) -> index
            %37 = krnl.load %alloc_8[%36] : memref<6xi1>
            %38 = arith.cmpi eq, %37, %false : i1
            scf.if %38 {
              %39 = krnl.load %alloc_3[%16#0, %36, %c0] : memref<1x6x4xf32>
              %40 = krnl.load %alloc_3[%16#0, %36, %c1] : memref<1x6x4xf32>
              %41 = krnl.load %alloc_3[%16#0, %36, %c2] : memref<1x6x4xf32>
              %42 = krnl.load %alloc_3[%16#0, %36, %c3] : memref<1x6x4xf32>
              %43 = arith.subf %31, %29 : f32
              %44 = arith.subf %30, %28 : f32
              %45 = arith.mulf %44, %43 : f32
              %46 = arith.subf %42, %40 : f32
              %47 = arith.subf %41, %39 : f32
              %48 = arith.mulf %47, %46 : f32
              %49 = arith.maxf %29, %40 : f32
              %50 = arith.maxf %28, %39 : f32
              %51 = arith.minf %31, %42 : f32
              %52 = arith.minf %30, %41 : f32
              %53 = arith.subf %51, %49 : f32
              %54 = arith.subf %52, %50 : f32
              %55 = arith.maxf %54, %cst_0 : f32
              %56 = arith.maxf %53, %cst_0 : f32
              %57 = arith.mulf %56, %55 : f32
              %58 = arith.addf %45, %48 : f32
              %59 = arith.subf %58, %57 : f32
              %60 = arith.addf %59, %cst : f32
              %61 = arith.divf %57, %60 : f32
              %62 = arith.cmpf oge, %61, %2 : f32
              scf.if %62 {
                krnl.store %true, %alloc_8[%36] : memref<6xi1>
              }
            }
          }
        }
      }
    }
    %14 = krnl.load %alloca_5[] : memref<index>
    %alloc_7 = memref.alloc(%14) {alignment = 16 : i64} : memref<?x3xi64>
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg5 = %c0 to %14, %15#1 -> %arg6 = %c0 to %c3){
      %16:2 = krnl.get_induction_var_value(%15#0, %15#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %17 = krnl.load %alloc_4[%16#0, %16#1] : memref<?x3xindex>
      %18 = arith.index_cast %17 : index to i64
      krnl.store %18, %alloc_7[%16#0, %16#1] : memref<?x3xi64>
    }
    return %alloc_7 : memref<?x3xi64>
  }
}


// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @test_nonmaxsuppression_unknown_dims(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<1xi64>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<?x3xi64> {
    %cst = arith.constant 9.99999993E-9 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %c-1 = arith.constant -1 : index
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %false = arith.constant false
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = krnl.load %arg2[%c0] : memref<1xi64>
    %1 = krnl.load %arg4[%c0] : memref<1xf32>
    %2 = krnl.load %arg3[%c0] : memref<1xf32>
    %dim = memref.dim %arg1, %c0 : memref<?x?x?xf32>
    %dim_2 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
    %dim_3 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    %alloca = memref.alloca() : memref<index>
    %3 = arith.index_cast %0 : i64 to index
    %4 = arith.minsi %3, %dim_3 : index
    krnl.store %4, %alloca[] : memref<index>
    %dim_4 = memref.dim %arg1, %c0 : memref<?x?x?xf32>
    %dim_5 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
    %dim_6 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    %alloca_7 = memref.alloca() : memref<index>
    %alloca_8 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_8[] : memref<index>
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg5 = %c0 to %dim_4, %5#1 -> %arg6 = %c0 to %dim_5){
      %16:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_7[] : memref<index>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %dim_6){
        %21 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %22 = krnl.load %arg1[%16#0, %16#1, %21] : memref<?x?x?xf32>
        %23 = arith.cmpf ogt, %22, %1 : f32
        %24 = krnl.load %alloca_7[] : memref<index>
        %25 = arith.addi %24, %c1 : index
        %26 = arith.select %23, %25, %24 : index
        krnl.store %26, %alloca_7[] : memref<index>
      }
      %18 = krnl.load %alloca_7[] : memref<index>
      %19 = krnl.load %alloca_8[] : memref<index>
      %20 = arith.maxsi %18, %19 : index
      krnl.store %20, %alloca_8[] : memref<index>
    }
    %6 = krnl.load %alloca[] : memref<index>
    %7 = krnl.load %alloca_8[] : memref<index>
    %8 = arith.minsi %6, %7 : index
    krnl.store %8, %alloca[] : memref<index>
    %9 = krnl.load %alloca[] : memref<index>
    %dim_9 = memref.dim %arg1, %c0 : memref<?x?x?xf32>
    %dim_10 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
    %dim_11 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
    %alloc = memref.alloc(%dim_9, %dim_10, %dim_11) {alignment = 16 : i64} : memref<?x?x?xindex>
    %10:3 = krnl.define_loops 3
    krnl.iterate(%10#0, %10#1, %10#2) with (%10#0 -> %arg5 = 0 to #map(%dim_9), %10#1 -> %arg6 = 0 to #map1(%dim_9, %dim_10), %10#2 -> %arg7 = 0 to #map2(%dim_9, %dim_10, %dim_11)){
      %16:3 = krnl.get_induction_var_value(%10#0, %10#1, %10#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      krnl.store %16#2, %alloc[%16#0, %16#1, %16#2] : memref<?x?x?xindex>
    }
    "krnl.call"(%alloc, %arg1, %c2_i64, %c0_i64) <{funcName = "omTensorSort", numOfOutput = 1 : si64}> : (memref<?x?x?xindex>, memref<?x?x?xf32>, i64, i64) -> ()
    %11 = arith.muli %dim, %dim_2 : index
    %12 = arith.muli %11, %9 : index
    %alloc_12 = memref.alloc(%12) {alignment = 16 : i64} : memref<?x3xindex>
    krnl.memset %alloc_12, %c-1 : memref<?x3xindex>
    %alloca_13 = memref.alloca() : memref<index>
    krnl.store %c0, %alloca_13[] : memref<index>
    %alloca_14 = memref.alloca() : memref<index>
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg5 = %c0 to %dim, %13#1 -> %arg6 = %c0 to %dim_2){
      %16:2 = krnl.get_induction_var_value(%13#0, %13#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.store %c0, %alloca_14[] : memref<index>
      %alloc_16 = memref.alloc(%dim_3) {alignment = 16 : i64} : memref<?xi1>
      krnl.memset %alloc_16, %false : memref<?xi1>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg7 = %c0 to %dim_3){
        %18 = krnl.get_induction_var_value(%17) : (!krnl.loop) -> index
        %19 = krnl.load %alloc[%16#0, %16#1, %18] : memref<?x?x?xindex>
        %20 = krnl.load %arg1[%16#0, %16#1, %19] : memref<?x?x?xf32>
        %21 = arith.cmpf ogt, %20, %1 : f32
        %22 = krnl.load %alloca_14[] : memref<index>
        %23 = arith.cmpi slt, %22, %9 : index
        %24 = krnl.load %alloc_16[%19] : memref<?xi1>
        %25 = arith.cmpi eq, %24, %false : i1
        %26 = arith.andi %21, %23 : i1
        %27 = arith.andi %26, %25 : i1
        scf.if %27 {
          %28 = krnl.load %arg0[%16#0, %19, %c0] : memref<?x?x?xf32>
          %29 = krnl.load %arg0[%16#0, %19, %c1] : memref<?x?x?xf32>
          %30 = krnl.load %arg0[%16#0, %19, %c2] : memref<?x?x?xf32>
          %31 = krnl.load %arg0[%16#0, %19, %c3] : memref<?x?x?xf32>
          %32 = krnl.load %alloca_13[] : memref<index>
          krnl.store %16#0, %alloc_12[%32, %c0] : memref<?x3xindex>
          krnl.store %16#1, %alloc_12[%32, %c1] : memref<?x3xindex>
          krnl.store %19, %alloc_12[%32, %c2] : memref<?x3xindex>
          %33 = arith.addi %22, %c1 : index
          krnl.store %33, %alloca_14[] : memref<index>
          %34 = arith.addi %32, %c1 : index
          krnl.store %34, %alloca_13[] : memref<index>
          %35 = krnl.define_loops 1
          krnl.iterate(%35) with (%35 -> %arg8 = %c0 to %dim_3){
            %36 = krnl.get_induction_var_value(%35) : (!krnl.loop) -> index
            %37 = krnl.load %alloc_16[%36] : memref<?xi1>
            %38 = arith.cmpi eq, %37, %false : i1
            scf.if %38 {
              %39 = krnl.load %arg0[%16#0, %36, %c0] : memref<?x?x?xf32>
              %40 = krnl.load %arg0[%16#0, %36, %c1] : memref<?x?x?xf32>
              %41 = krnl.load %arg0[%16#0, %36, %c2] : memref<?x?x?xf32>
              %42 = krnl.load %arg0[%16#0, %36, %c3] : memref<?x?x?xf32>
              %43 = arith.divf %30, %cst_1 : f32
              %44 = arith.subf %28, %43 : f32
              %45 = arith.divf %30, %cst_1 : f32
              %46 = arith.addf %28, %45 : f32
              %47 = arith.divf %31, %cst_1 : f32
              %48 = arith.subf %29, %47 : f32
              %49 = arith.divf %31, %cst_1 : f32
              %50 = arith.addf %29, %49 : f32
              %51 = arith.divf %42, %cst_1 : f32
              %52 = arith.subf %40, %51 : f32
              %53 = arith.divf %42, %cst_1 : f32
              %54 = arith.addf %40, %53 : f32
              %55 = arith.divf %41, %cst_1 : f32
              %56 = arith.subf %39, %55 : f32
              %57 = arith.divf %41, %cst_1 : f32
              %58 = arith.addf %39, %57 : f32
              %59 = arith.mulf %31, %30 : f32
              %60 = arith.mulf %42, %41 : f32
              %61 = arith.maxf %44, %56 : f32
              %62 = arith.maxf %48, %52 : f32
              %63 = arith.minf %46, %58 : f32
              %64 = arith.minf %50, %54 : f32
              %65 = arith.subf %63, %61 : f32
              %66 = arith.subf %64, %62 : f32
              %67 = arith.maxf %66, %cst_0 : f32
              %68 = arith.maxf %65, %cst_0 : f32
              %69 = arith.mulf %68, %67 : f32
              %70 = arith.addf %59, %60 : f32
              %71 = arith.subf %70, %69 : f32
              %72 = arith.addf %71, %cst : f32
              %73 = arith.divf %69, %72 : f32
              %74 = arith.cmpf oge, %73, %2 : f32
              scf.if %74 {
                krnl.store %true, %alloc_16[%36] : memref<?xi1>
              }
            }
          }
        }
      }
    }
    %14 = krnl.load %alloca_13[] : memref<index>
    %alloc_15 = memref.alloc(%14) {alignment = 16 : i64} : memref<?x3xi64>
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg5 = %c0 to %14, %15#1 -> %arg6 = %c0 to %c3){
      %16:2 = krnl.get_induction_var_value(%15#0, %15#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %17 = krnl.load %alloc_12[%16#0, %16#1] : memref<?x3xindex>
      %18 = arith.index_cast %17 : index to i64
      krnl.store %18, %alloc_15[%16#0, %16#1] : memref<?x3xi64>
    }
    return %alloc_15 : memref<?x3xi64>
  }
}

