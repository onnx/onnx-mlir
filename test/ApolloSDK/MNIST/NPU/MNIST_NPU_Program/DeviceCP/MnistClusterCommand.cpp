//
// Copyright (C) Microsoft Corporation. All rights reserved.
//

#include "MnistClusterCommand.h"
#include "Nepal/ArrayRef.h"
#include "tcp_driver.tcp.h"

#include <Commands/CommandRuntime.h>
#include <TrainwaveClusterSDK.h>

#include <KernelParams.h>

#pragma warning(push, 0)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

// For Visual Studio
#pragma warning(disable : 4100)

// Register the ClusterCP interface with the runtime
REGISTER_CLUSTERCP_INTERFACE(Mnist::ClusterCP::Interface)

namespace npl = Apollo::Nepal;
namespace npli = Apollo::Nepal::Internal;
namespace twf = Trainwave::FirmwareSDK;
namespace twfCluster = Trainwave::FirmwareSDK::Cluster;
namespace mmTileCP = MaiaCompiler::GeneratedCode;
namespace twrProgram = Trainwave::FirmwareRT::Program;

namespace Mnist::ClusterCP {
using DMemArray2D = npl::ArrayRef<2, npl::MemoryKind::DeviceMem,
    npl::ElementDataType::BFloat16, npl::FormatKind::Tile>;

// Relays the PipelinedMatMul task for the TileCP to execute.
// Waits for the TileCP to finish before returning.
void MnistClusterCommand::Execute(const MnistClusterCommand::Arguments &args) {
  const auto NativeMatrixDim =
      Trainwave::FirmwareRT::GetHWInfo().GetNativeMatrixDim();

  npl::NotificationList done =
      npl::BuildNotificationList(npl::NotifConsumerKind::ClusterCP);

  auto input1 =
      reinterpret_cast<KernelParams::ArrayRef const *>(args.params.data);
  auto input2 = input1 + 1;
  auto input3 = input1 + 2;
  auto input4 = input1 + 3;
  auto input5 = input1 + 4;
  auto output = input1 + 5;

  DMemArray2D arg0({256, 256}, input1->dramAddress);
  DMemArray2D arg1({256, 256}, input2->dramAddress);
  DMemArray2D arg2({256, 256}, input3->dramAddress);
  DMemArray2D arg3({256, 256}, input4->dramAddress);
  DMemArray2D arg4({256, 256}, input5->dramAddress);
  DMemArray2D arg5({256, 256}, output->dramAddress); // Output

  mmTileCP::main_graph::Arguments cmdArgs = {
      arg0, arg1, arg2, arg3, arg4, arg5, done};

  const Apollo::Primitives::SemId controlSemId =
      twf::SemManager::MapClusterSemOffset(
          twf::SemManager::GetNumClusterSem() - 1);

  DebugPuts("DeviceCP: Calling TileCP...\n");
  twfCluster::template CallTileCPAsync<mmTileCP::main_graph>(
      cmdArgs, controlSemId);

  DebugPuts("DeviceCP: Waiting for TileCP to finish...\n");
  done[0].Wait();

  DebugPuts("DeviceCP: Finishing TileCP...\n");

  // Write funcId to OutputCtrlMsgFifo to signal firmware complete
  const uint32_t funcId = twf::GetCommandID<MnistClusterCommand, Interface>();
  twrProgram::WriteMessageWord(funcId);
}
} // namespace Mnist::ClusterCP

#pragma clang diagnostic pop
#pragma warning(pop)
