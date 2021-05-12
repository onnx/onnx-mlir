//
// Copyright (C) Microsoft Corporation. All rights reserved.
//
#pragma once
#include "Commands/CommandRuntime.h"
#include <KernelParams.h>
#include <stdint.h>
namespace Mnist::ClusterCP {
// Runs no-op firmware which enqueues commands for each accelerator
// which utilizes the Nios DMA to move commands from Nios DMem to each
// child processor's command queue
struct MnistClusterCommand {
#pragma pack(push, 1)
  struct Arguments {
    KernelParams params;
  };
#pragma pack(pop)

  static void Execute(const Arguments &);
};

using Interface = Trainwave::FirmwareSDK::CommandList<MnistClusterCommand>;
} // namespace Mnist::ClusterCP