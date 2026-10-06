; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@om_compilation_info_json_after = internal constant [3 x i8] c"{}\00"
@"om_Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A_after" = internal constant [70 x i8] c"Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A", align 16
@"om_Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A_after" = internal constant [70 x i8] c"Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A", align 16
@"om_Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A_after" = internal constant [70 x i8] c"Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A", align 16
@"om_Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A_after" = internal constant [70 x i8] c"Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A", align 16
@"om_Wrong rank for the input 0: expect 4, but got %lld\0A_after" = internal constant [51 x i8] c"Wrong rank for the input 0: expect 4, but got %lld\0A", align 16
@"om_Wrong data type for the input 0: expect f32\0A_after" = internal constant [44 x i8] c"Wrong data type for the input 0: expect f32\0A", align 16
@"om_Wrong number of input tensors: expect 1, but got %lld\0A_after" = internal constant [54 x i8] c"Wrong number of input tensors: expect 1, but got %lld\0A", align 16
@_entry_point_1_after = constant [21 x i8] c"run_main_graph_after\00"
@_entry_point_1_in_sig_after = constant [75 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 3 , 4] , \22name\22 : \22input_0\22 }\0A\0A]\00"
@_entry_point_1_out_sig_after = constant [75 x i8] c"[   { \22type\22 : \22i64\22 , \22dims\22 : [1 , 4 , 2 , 3] , \22name\22 : \22output_0\22 }\0A\0A]\00"
@_entry_point_0_after = constant [15 x i8] c"run_main_graph\00"
@_entry_point_0_in_sig_after = constant [75 x i8] c"[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 3 , 4] , \22name\22 : \22input_0\22 }\0A\0A]\00"
@_entry_point_0_out_sig_after = constant [75 x i8] c"[   { \22type\22 : \22i64\22 , \22dims\22 : [1 , 4 , 2 , 3] , \22name\22 : \22output_0\22 }\0A\0A]\00"
@_entry_point_arrays_after = internal constant [3 x ptr] [ptr @_entry_point_0_after, ptr @_entry_point_1_after, ptr null]

declare i32 @strncmp(ptr, ptr, i64)

declare void @printf(ptr, ...)

declare ptr @__errno_location()

declare void @omGetExternalConstantAddr(ptr, ptr, i64)

declare i1 @omUnloadConstantData(ptr, i64)

declare i1 @omLoadConstantData(ptr, ptr, i64, i64)

declare i64 @omTensorListGetSize(ptr)

declare void @omTensorPrint(ptr, ptr)

declare ptr @omTensorListGetOmtArray(ptr)

declare void @omTensorSetDataType(ptr, i64)

declare i64 @omTensorGetDataType(ptr)

declare ptr @omTensorGetStrides(ptr)

declare ptr @omTensorGetShape(ptr)

declare i64 @omTensorGetRank(ptr)

declare void @omTensorSetDataPtr(ptr, i64, ptr, ptr)

declare ptr @omTensorGetDataPtr(ptr)

declare void @omTensorDestroy(ptr)

declare ptr @omTensorCreateUntyped(i64)

declare ptr @omTensorListCreate(ptr, i64)

declare void @free(ptr)

declare ptr @malloc(i64)

define { ptr, ptr, i64, [4 x i64], [4 x i64] } @main_graph_after(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = call ptr @malloc(i64 112)
  %13 = ptrtoint ptr %12 to i64
  %14 = add i64 %13, 15
  %15 = urem i64 %14, 16
  %16 = sub i64 %14, %15
  %17 = inttoptr i64 %16 to ptr
  br label %18

18:                                               ; preds = %49, %11
  %19 = phi i64 [ %50, %49 ], [ 0, %11 ]
  %20 = icmp slt i64 %19, 3
  br i1 %20, label %21, label %51

21:                                               ; preds = %18
  br label %22

22:                                               ; preds = %25, %21
  %23 = phi i64 [ %48, %25 ], [ 0, %21 ]
  %24 = icmp slt i64 %23, 4
  br i1 %24, label %25, label %49

25:                                               ; preds = %22
  %26 = mul nuw nsw i64 %19, 4
  %27 = add nuw nsw i64 0, %26
  %28 = add nuw nsw i64 %27, %23
  %29 = getelementptr inbounds nuw float, ptr %1, i64 %28
  %30 = load float, ptr %29, align 4
  %31 = mul nuw nsw i64 %19, 8
  %32 = add nuw nsw i64 0, %31
  %33 = mul nuw nsw i64 %23, 2
  %34 = add nuw nsw i64 %32, %33
  %35 = add nuw nsw i64 %34, 0
  %36 = getelementptr inbounds nuw float, ptr %17, i64 %35
  store float %30, ptr %36, align 4
  %37 = mul nuw nsw i64 %19, 4
  %38 = add nuw nsw i64 12, %37
  %39 = add nuw nsw i64 %38, %23
  %40 = getelementptr inbounds nuw float, ptr %1, i64 %39
  %41 = load float, ptr %40, align 4
  %42 = mul nuw nsw i64 %19, 8
  %43 = add nuw nsw i64 0, %42
  %44 = mul nuw nsw i64 %23, 2
  %45 = add nuw nsw i64 %43, %44
  %46 = add nuw nsw i64 %45, 1
  %47 = getelementptr inbounds nuw float, ptr %17, i64 %46
  store float %41, ptr %47, align 4
  %48 = add i64 %23, 1
  br label %22

49:                                               ; preds = %22
  %50 = add i64 %19, 1
  br label %18

51:                                               ; preds = %18
  %52 = call ptr @malloc(i64 208)
  %53 = ptrtoint ptr %52 to i64
  %54 = add i64 %53, 15
  %55 = urem i64 %54, 16
  %56 = sub i64 %54, %55
  %57 = inttoptr i64 %56 to ptr
  br label %58

58:                                               ; preds = %87, %51
  %59 = phi i64 [ %88, %87 ], [ 0, %51 ]
  %60 = icmp slt i64 %59, 3
  br i1 %60, label %61, label %89

61:                                               ; preds = %58
  br label %62

62:                                               ; preds = %85, %61
  %63 = phi i64 [ %86, %85 ], [ 0, %61 ]
  %64 = icmp slt i64 %63, 4
  br i1 %64, label %65, label %87

65:                                               ; preds = %62
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i64 [ %84, %69 ], [ 0, %65 ]
  %68 = icmp slt i64 %67, 2
  br i1 %68, label %69, label %85

69:                                               ; preds = %66
  %70 = mul nuw nsw i64 %59, 8
  %71 = add nuw nsw i64 0, %70
  %72 = mul nuw nsw i64 %63, 2
  %73 = add nuw nsw i64 %71, %72
  %74 = add nuw nsw i64 %73, %67
  %75 = getelementptr inbounds nuw float, ptr %17, i64 %74
  %76 = load float, ptr %75, align 4
  %77 = fptosi float %76 to i64
  %78 = mul nuw nsw i64 %59, 8
  %79 = add nuw nsw i64 0, %78
  %80 = mul nuw nsw i64 %63, 2
  %81 = add nuw nsw i64 %79, %80
  %82 = add nuw nsw i64 %81, %67
  %83 = getelementptr inbounds nuw i64, ptr %57, i64 %82
  store i64 %77, ptr %83, align 8
  %84 = add i64 %67, 1
  br label %66

85:                                               ; preds = %66
  %86 = add i64 %63, 1
  br label %62

87:                                               ; preds = %62
  %88 = add i64 %59, 1
  br label %58

89:                                               ; preds = %58
  call void @free(ptr %12)
  %90 = call ptr @malloc(i64 208)
  %91 = ptrtoint ptr %90 to i64
  %92 = add i64 %91, 15
  %93 = urem i64 %92, 16
  %94 = sub i64 %92, %93
  %95 = inttoptr i64 %94 to ptr
  %96 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } poison, ptr %90, 0
  %97 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %96, ptr %95, 1
  %98 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %97, i64 0, 2
  %99 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %98, i64 1, 3, 0
  %100 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %99, i64 4, 3, 1
  %101 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %100, i64 2, 3, 2
  %102 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %101, i64 3, 3, 3
  %103 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %102, i64 24, 4, 0
  %104 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %103, i64 6, 4, 1
  %105 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %104, i64 3, 4, 2
  %106 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %105, i64 1, 4, 3
  br label %107

107:                                              ; preds = %149, %89
  %108 = phi i64 [ %150, %149 ], [ 0, %89 ]
  %109 = icmp slt i64 %108, 4
  br i1 %109, label %110, label %151

110:                                              ; preds = %107
  br label %111

111:                                              ; preds = %114, %110
  %112 = phi i64 [ %148, %114 ], [ 0, %110 ]
  %113 = icmp slt i64 %112, 2
  br i1 %113, label %114, label %149

114:                                              ; preds = %111
  %115 = mul nuw nsw i64 %108, 2
  %116 = add nuw nsw i64 0, %115
  %117 = add nuw nsw i64 %116, %112
  %118 = getelementptr inbounds nuw i64, ptr %57, i64 %117
  %119 = load i64, ptr %118, align 8
  %120 = mul nuw nsw i64 %108, 6
  %121 = add nuw nsw i64 0, %120
  %122 = mul nuw nsw i64 %112, 3
  %123 = add nuw nsw i64 %121, %122
  %124 = add nuw nsw i64 %123, 0
  %125 = getelementptr inbounds nuw i64, ptr %95, i64 %124
  store i64 %119, ptr %125, align 8
  %126 = mul nuw nsw i64 %108, 2
  %127 = add nuw nsw i64 8, %126
  %128 = add nuw nsw i64 %127, %112
  %129 = getelementptr inbounds nuw i64, ptr %57, i64 %128
  %130 = load i64, ptr %129, align 8
  %131 = mul nuw nsw i64 %108, 6
  %132 = add nuw nsw i64 0, %131
  %133 = mul nuw nsw i64 %112, 3
  %134 = add nuw nsw i64 %132, %133
  %135 = add nuw nsw i64 %134, 1
  %136 = getelementptr inbounds nuw i64, ptr %95, i64 %135
  store i64 %130, ptr %136, align 8
  %137 = mul nuw nsw i64 %108, 2
  %138 = add nuw nsw i64 16, %137
  %139 = add nuw nsw i64 %138, %112
  %140 = getelementptr inbounds nuw i64, ptr %57, i64 %139
  %141 = load i64, ptr %140, align 8
  %142 = mul nuw nsw i64 %108, 6
  %143 = add nuw nsw i64 0, %142
  %144 = mul nuw nsw i64 %112, 3
  %145 = add nuw nsw i64 %143, %144
  %146 = add nuw nsw i64 %145, 2
  %147 = getelementptr inbounds nuw i64, ptr %95, i64 %146
  store i64 %141, ptr %147, align 8
  %148 = add i64 %112, 1
  br label %111

149:                                              ; preds = %111
  %150 = add i64 %108, 1
  br label %107

151:                                              ; preds = %107
  call void @free(ptr %52)
  ret { ptr, ptr, i64, [4 x i64], [4 x i64] } %106
}

define void @_mlir_ciface_main_graph_after(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 2
  %10 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 3, 3
  %11 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 1
  %13 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 2
  %14 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %3, 4, 3
  %15 = call { ptr, ptr, i64, [4 x i64], [4 x i64] } @main_graph_after(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14)
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %15, ptr %0, align 8
  ret void
}

define ptr @run_main_graph_after(ptr %0) {
  %2 = call i64 @omTensorListGetSize(ptr %0)
  %3 = icmp ne i64 1, %2
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void (ptr, ...) @printf(ptr @"om_Wrong number of input tensors: expect 1, but got %lld\0A_after", i64 %2)
  %5 = call ptr @__errno_location()
  store i32 22, ptr %5, align 4
  ret ptr null

6:                                                ; preds = %1
  %7 = call ptr @omTensorListGetOmtArray(ptr %0)
  %8 = load ptr, ptr %7, align 8
  %9 = call i64 @omTensorGetDataType(ptr %8)
  %10 = icmp ne i64 1, %9
  br i1 %10, label %11, label %13

11:                                               ; preds = %6
  call void (ptr, ...) @printf(ptr @"om_Wrong data type for the input 0: expect f32\0A_after")
  %12 = call ptr @__errno_location()
  store i32 22, ptr %12, align 4
  ret ptr null

13:                                               ; preds = %6
  %14 = call i64 @omTensorGetRank(ptr %8)
  %15 = icmp ne i64 4, %14
  br i1 %15, label %16, label %18

16:                                               ; preds = %13
  call void (ptr, ...) @printf(ptr @"om_Wrong rank for the input 0: expect 4, but got %lld\0A_after", i64 %14)
  %17 = call ptr @__errno_location()
  store i32 22, ptr %17, align 4
  ret ptr null

18:                                               ; preds = %13
  %19 = call ptr @omTensorGetShape(ptr %8)
  %20 = load i64, ptr %19, align 8
  %21 = icmp ne i64 1, %20
  br i1 %21, label %22, label %24

22:                                               ; preds = %18
  call void (ptr, ...) @printf(ptr @"om_Wrong size for the dimension 0 of the input 0: expect 1, but got %lld\0A_after", i64 %20)
  %23 = call ptr @__errno_location()
  store i32 22, ptr %23, align 4
  ret ptr null

24:                                               ; preds = %18
  %25 = getelementptr i64, ptr %19, i32 1
  %26 = load i64, ptr %25, align 8
  %27 = icmp ne i64 2, %26
  br i1 %27, label %28, label %30

28:                                               ; preds = %24
  call void (ptr, ...) @printf(ptr @"om_Wrong size for the dimension 1 of the input 0: expect 2, but got %lld\0A_after", i64 %26)
  %29 = call ptr @__errno_location()
  store i32 22, ptr %29, align 4
  ret ptr null

30:                                               ; preds = %24
  %31 = getelementptr i64, ptr %19, i32 2
  %32 = load i64, ptr %31, align 8
  %33 = icmp ne i64 3, %32
  br i1 %33, label %34, label %36

34:                                               ; preds = %30
  call void (ptr, ...) @printf(ptr @"om_Wrong size for the dimension 2 of the input 0: expect 3, but got %lld\0A_after", i64 %32)
  %35 = call ptr @__errno_location()
  store i32 22, ptr %35, align 4
  ret ptr null

36:                                               ; preds = %30
  %37 = getelementptr i64, ptr %19, i32 3
  %38 = load i64, ptr %37, align 8
  %39 = icmp ne i64 4, %38
  br i1 %39, label %40, label %42

40:                                               ; preds = %36
  call void (ptr, ...) @printf(ptr @"om_Wrong size for the dimension 3 of the input 0: expect 4, but got %lld\0A_after", i64 %38)
  %41 = call ptr @__errno_location()
  store i32 22, ptr %41, align 4
  ret ptr null

42:                                               ; preds = %36
  %43 = call ptr @omTensorListGetOmtArray(ptr %0)
  %44 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  %45 = load ptr, ptr %43, align 8
  %46 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  %47 = call ptr @omTensorGetDataPtr(ptr %45)
  %48 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %47, 0
  %49 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %48, ptr %47, 1
  %50 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, i64 0, 2
  %51 = call ptr @omTensorGetShape(ptr %45)
  %52 = call ptr @omTensorGetStrides(ptr %45)
  %53 = load i64, ptr %51, align 8
  %54 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %50, i64 %53, 3, 0
  %55 = load i64, ptr %52, align 8
  %56 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %54, i64 %55, 4, 0
  %57 = getelementptr i64, ptr %51, i32 1
  %58 = load i64, ptr %57, align 8
  %59 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %56, i64 %58, 3, 1
  %60 = getelementptr i64, ptr %52, i32 1
  %61 = load i64, ptr %60, align 8
  %62 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %59, i64 %61, 4, 1
  %63 = getelementptr i64, ptr %51, i32 2
  %64 = load i64, ptr %63, align 8
  %65 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %62, i64 %64, 3, 2
  %66 = getelementptr i64, ptr %52, i32 2
  %67 = load i64, ptr %66, align 8
  %68 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %65, i64 %67, 4, 2
  %69 = getelementptr i64, ptr %51, i32 3
  %70 = load i64, ptr %69, align 8
  %71 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %68, i64 %70, 3, 3
  %72 = getelementptr i64, ptr %52, i32 3
  %73 = load i64, ptr %72, align 8
  %74 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %71, i64 %73, 4, 3
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %74, ptr %46, align 8
  call void @_mlir_ciface_main_graph_after(ptr %44, ptr %46)
  %75 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %44, align 8
  %76 = alloca ptr, i64 1, align 8
  %77 = call ptr @omTensorCreateUntyped(i64 4)
  %78 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 0
  %79 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 1
  call void @omTensorSetDataPtr(ptr %77, i64 1, ptr %78, ptr %79)
  call void @omTensorSetDataType(ptr %77, i64 7)
  %80 = call ptr @omTensorGetShape(ptr %77)
  %81 = call ptr @omTensorGetStrides(ptr %77)
  %82 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 3, 0
  store i64 %82, ptr %80, align 8
  %83 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 4, 0
  store i64 %83, ptr %81, align 8
  %84 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 3, 1
  %85 = getelementptr i64, ptr %80, i32 1
  store i64 %84, ptr %85, align 8
  %86 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 4, 1
  %87 = getelementptr i64, ptr %81, i32 1
  store i64 %86, ptr %87, align 8
  %88 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 3, 2
  %89 = getelementptr i64, ptr %80, i32 2
  store i64 %88, ptr %89, align 8
  %90 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 4, 2
  %91 = getelementptr i64, ptr %81, i32 2
  store i64 %90, ptr %91, align 8
  %92 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 3, 3
  %93 = getelementptr i64, ptr %80, i32 3
  store i64 %92, ptr %93, align 8
  %94 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, 4, 3
  %95 = getelementptr i64, ptr %81, i32 3
  store i64 %94, ptr %95, align 8
  store ptr %77, ptr %76, align 8
  %96 = call ptr @omTensorListCreate(ptr %76, i64 1)
  ret ptr %96
}

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_after(ptr %0)
  ret ptr %2
}

define ptr @omQueryEntryPoints_after(ptr %0) {
  %2 = icmp ne ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  store i64 2, ptr %0, align 8
  br label %4

4:                                                ; preds = %3, %1
  ret ptr @_entry_point_arrays_after
}

define ptr @omQueryEntryPoints(ptr %0) {
  %2 = call ptr @omQueryEntryPoints_after(ptr %0)
  ret ptr %2
}

define ptr @omInputSignature_after(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_after, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_in_sig_after

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_after, i64 21)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_in_sig_after

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omInputSignature(ptr %0) {
  %2 = call ptr @omInputSignature_after(ptr %0)
  ret ptr %2
}

define ptr @omOutputSignature_after(ptr %0) {
  %2 = call i32 @strncmp(ptr %0, ptr @_entry_point_0_after, i64 15)
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  ret ptr @_entry_point_0_out_sig_after

5:                                                ; preds = %1
  %6 = call i32 @strncmp(ptr %0, ptr @_entry_point_1_after, i64 21)
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  ret ptr @_entry_point_1_out_sig_after

9:                                                ; preds = %5
  ret ptr null
}

define ptr @omOutputSignature(ptr %0) {
  %2 = call ptr @omOutputSignature_after(ptr %0)
  ret ptr %2
}

define ptr @omCompilationInfo_after() {
  ret ptr @om_compilation_info_json_after
}

define ptr @omCompilationInfo() {
  %1 = call ptr @omCompilationInfo_after()
  ret ptr %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
