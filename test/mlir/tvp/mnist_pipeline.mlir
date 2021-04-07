// RUN: onnx-mlir --EmitApollo %s -o %t 
// RUN: FileCheck %s --input-file %t.s -check-prefix=CHECK-ASM
// RUN: FileCheck %s --input-file %t.tcp.h -check-prefix=CHECK-TCP-H
// RUN: FileCheck %s --input-file %t.tvp.h -check-prefix=CHECK-TVP-H
// RUN: FileCheck %s --input-file %t.tcp.cpp -check-prefix=CHECK-TCP-CPP
// RUN: FileCheck %s --input-file %t.tvp.cpp -check-prefix=CHECK-TVP-CPP

// This is the template for e2e test for MNIST pipeline.
//   - check that each file contains the appropriate prologue
//   - check that the TCP implementation code (.tcp.cpp) contains an implementation (further checking is NYI)
//   - check that the TVP implementation code (.s) contains an implementation (further checking is NYI)

// CHECK-TCP-H: //
// CHECK-TCP-H: // Copyright (C) Microsoft Corporation. All rights reserved.
// CHECK-TCP-H: //
// CHECK-TCP-H: #pragma once
// CHECK-TCP-H: #include "Nepal/ArrayRef.h"
// CHECK-TCP-H: #include "Notification.h"
// CHECK-TCP-H: #include "Commands/CommandInterfaces.h"
// CHECK-TCP-H: #include "Commands/CommandList.h"
// CHECK-TCP-H: namespace [[NS:.*]] {
// CHECK-TCP-H:    namespace npl = Apollo::Nepal;
// CHECK-TCP-H:    template <npl::MemoryKind TMemory, npl::ElementDataType TDataType>
// CHECK-TCP-H:    using Array2D = npl::ArrayRef<2, TMemory, TDataType, Apollo::Nepal::FormatKind::Tile>;
// CHECK-TCP-H:    struct main_graph {
// CHECK-TCP-H:       struct Arguments {
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          Array2D<npl::MemoryKind::DeviceMem, npl::ElementDataType::BFloat16> {{.*}};
// CHECK-TCP-H:          npl::NotificationList {{.*}};
// CHECK-TCP-H:       };
// CHECK-TCP-H:       static void Execute(const Arguments&);
// CHECK-TCP-H:    };
// CHECK-TCP-H: }
// CHECK-TCP-H: namespace Trainwave::FirmwareSDK {
// CHECK-TCP-H:    template <> struct TileCPInterface<0> {
// CHECK-TCP-H:       using type = Trainwave::FirmwareSDK::CommandList<[[NS]]::main_graph>;
// CHECK-TCP-H:    };
// CHECK-TCP-H: }

// CHECK-TVP-H: //
// CHECK-TVP-H: // Copyright (C) Microsoft Corporation. All rights reserved.
// CHECK-TVP-H: //
// CHECK-TVP-H: #pragma once
// CHECK-TVP-H: namespace [[NS:.*]] {
// CHECK-TVP-H:    struct Relu {
// CHECK-TVP-H:       struct Arguments {
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:       };
// CHECK-TVP-H:       static void Execute(const Arguments&);
// CHECK-TVP-H:    };
// CHECK-TVP-H:    struct Add_0 {
// CHECK-TVP-H:       struct Arguments {
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:       };
// CHECK-TVP-H:       static void Execute(const Arguments&);
// CHECK-TVP-H:    };
// CHECK-TVP-H:    struct Add {
// CHECK-TVP-H:       struct Arguments {
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:       };
// CHECK-TVP-H:       static void Execute(const Arguments&);
// CHECK-TVP-H:    };
// CHECK-TVP-H:    struct Mul {
// CHECK-TVP-H:       struct Arguments {
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:          int32_t {{.*}};
// CHECK-TVP-H:       };
// CHECK-TVP-H:       static void Execute(const Arguments&);
// CHECK-TVP-H:    };
// CHECK-TVP-H: } 
// CHECK-TVP-H: namespace Trainwave::FirmwareSDK {
// CHECK-TVP-H:    struct TVPInterface {
// CHECK-TVP-H:       using type = Trainwave::FirmwareSDK::CommandList<[[NS]]::Relu, [[NS]]::Add_0, [[NS]]::Add, [[NS]]::Mul>;
// CHECK-TPV-H:    };
// CHECK-TVP-H: }

// CHECK-TVP-CPP: //
// CHECK-TVP-CPP: // Copyright (C) Microsoft Corporation. All rights reserved.
// CHECK-TVP-CPP: //
// CHECK-TVP-CPP: #include "Commands/CommandRuntime.h"
// CHECK-TVP-CPP: using int32_t = int;
// CHECK-TVP-CPP: #include "{{.*}}.tvp.h"
// CHECK-TVP-CPP: REGISTER_TVP_INTERFACE(Trainwave::FirmwareSDK::TVPInterface::type)
// CHECK-TVP-CPP: extern "C" void Execute(const int, const void*, const int);
// CHECK-TVP-CPP: namespace [[NS:.*]] {
// CHECK-TVP-CPP:    void Relu::Execute(const Arguments& args) {
// CHECK-TVP-CPP:       ::Execute(2, &args, 1 /* dummy */);
// CHECK-TVP-CPP:    }
// CHECK-TVP-CPP:    void Add_0::Execute(const Arguments& args) {
// CHECK-TVP-CPP:       ::Execute(3, &args, 1 /* dummy */);
// CHECK-TVP-CPP:    }
// CHECK-TVP-CPP:    void Add::Execute(const Arguments& args) {
// CHECK-TVP-CPP:       ::Execute(1, &args, 1 /* dummy */);
// CHECK-TVP-CPP:    }
// CHECK-TVP-CPP:    void Mul::Execute(const Arguments& args) {
// CHECK-TVP-CPP:       ::Execute(0, &args, 1 /* dummy */);
// CHECK-TVP-CPP:    }
// CHECK-TVP-CPP: }


// CHECK-TCP-CPP: //
// CHECK-TCP-CPP: // Copyright (C) Microsoft Corporation. All rights reserved.
// CHECK-TCP-CPP: //
// CHECK-TCP-CPP: #include "Commands/CommandRuntime.h"
// CHECK-TCP-CPP: #include "TileCP/TrainwaveTileSDK.h"
// CHECK-TCP-CPP: #include "TtuCommands/MatMulCommandInternal.h"
// CHECK-TCP-CPP: #include "TtuCommands/MatMulNoBiasCommand.h"
// CHECK-TCP-CPP: #include "[[FILE:.*]].tcp.h"
// CHECK-TCP-CPP: #include "[[FILE]].tvp.h"
// CHECK-TCP-CPP: REGISTER_TILECP_INTERFACE(Trainwave::FirmwareSDK::TileCPInterface<0>::type)
// CHECK-TCP-CPP: namespace [[NS:.*]] {
// CHECK-TCP-CPP:    namespace npli = Apollo::Nepal::Internal;
// CHECK-TCP-CPP:    namespace nplt = Apollo::Nepal::Tile;
// CHECK-TCP-CPP:    using MatMulCommand = npl::TtuCommands::MatMulNoBiasCommand;
// CHECK-TCP-CPP: }
// CHECK-TCP-CPP: namespace [[NS]] {
// CHECK-TCP-CPP: void main_graph::Execute(const Arguments &args)

// CHECK-ASM: Execute:

module  {
  func @main_graph(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x256xbf16>, %arg4: tensor<256xbf16>) -> tensor<256x256xbf16> attributes {input_names = ["X", "weight_0", "bias_0", "weight_1", "bias_1"], output_names = ["add_1_output"]} {
    %0 = "onnx.Constant"() {value = dense<-1.184680e-38> : tensor<bf16>} : () -> tensor<bf16>
    %1 = "onnx.Mul"(%arg0, %0) : (tensor<256x1024xbf16>, tensor<bf16>) -> tensor<*xbf16>
    %2 = "onnx.MatMul"(%1, %arg1) : (tensor<*xbf16>, tensor<1024x256xbf16>) -> tensor<*xbf16>
    %3 = "onnx.Add"(%2, %arg2) : (tensor<*xbf16>, tensor<256xbf16>) -> tensor<*xbf16>
    %4 = "onnx.Relu"(%3) : (tensor<*xbf16>) -> tensor<*xbf16>
    %5 = "onnx.MatMul"(%4, %arg3) : (tensor<*xbf16>, tensor<256x256xbf16>) -> tensor<*xbf16>
    %6 = "onnx.Add"(%5, %arg4) : (tensor<*xbf16>, tensor<256xbf16>) -> tensor<256x256xbf16>
    return %6 : tensor<256x256xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 5 : i32, numOutputs = 1 : i32, signature = "[    ]"} : () -> ()
}
