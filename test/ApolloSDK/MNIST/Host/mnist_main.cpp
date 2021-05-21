#include "tcp_driver.dcp.h"

#include <Commands/CommandMappingUtils.h>
#include <ControlMessage.h>
#include <KernelParams.h>
#include <Runtime/Device.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace twr = Trainwave::Runtime;
namespace twe = Trainwave::Emulator;

template <typename T, typename U>
gsl::span<T> AsSpan(U &obj) {
  static_assert(sizeof(U) % sizeof(T) == 0);
  return gsl::span<T>(reinterpret_cast<T *>(&obj), sizeof(U) / sizeof(T));
}

MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments
CreateKernelParams(std::vector<KernelParams::ArrayRef> &kernelInputs,
    std::vector<KernelParams::ArrayRef> &kernelOutputs) {
  MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments args{};
  args.params.numInputTensors = gsl::narrow<uint8_t>(kernelInputs.size());
  args.params.numOutputTensors = gsl::narrow<uint8_t>(kernelOutputs.size());
  gsl::span<uint32_t> buffer(args.params.data);

  auto it = buffer.begin();
  for (auto const &ref : kernelInputs) {
    auto src = AsSpan<const uint32_t>(ref);
    it = std::copy(src.begin(), src.end(), it);
  }
  for (auto const &ref : kernelOutputs) {
    auto src = AsSpan<const uint32_t>(ref);
    it = std::copy(src.begin(), src.end(), it);
  }
  return args;
}

int main(int argc, char *argv[]) {
  std::cerr << "Host: Starting" << std::endl;

  using SKU = SkuTraits<Sku_NpuTransformersSP>;
  using BFloat16Type = typename SKU::BFloat16Type;

  // Load the NPU Program.
  auto options = DeviceOptions();
  options.skuName = SKU::Name;
  auto device = twr::Device::CreateEmulatorDevice(options);

  std::cout << "Host: Loading NPU Program: " << std::string(argv[1])
            << std::endl;

  device.LoadNPUProgram(std::string(argv[1]));

  constexpr auto NativeMatrixDim = SKU::NativeMatrixDim0;

  KernelParams::ArrayRef::DimsType arg0_input_Dims = {256, 256};
  KernelParams::ArrayRef::DimsType arg1_weight0_Dims = {256, 256};
  KernelParams::ArrayRef::DimsType arg2_bias0_Dims = {256, 256};
  KernelParams::ArrayRef::DimsType arg3_weight1_Dims = {256, 256};
  KernelParams::ArrayRef::DimsType arg4_bias1_Dims = {256, 256};
  KernelParams::ArrayRef::DimsType arg5_output_Dims = {256, 256};

  std::vector<KernelParams::ArrayRef> kernelInputs;
  std::vector<KernelParams::ArrayRef> kernelOutputs;

  std::vector<bfloat16_t> dataInput(
      arg0_input_Dims[0] * arg0_input_Dims[1], 1.f);
  std::vector<bfloat16_t> dataWeight0(
      arg1_weight0_Dims[0] * arg1_weight0_Dims[1], 1.f);
  std::vector<bfloat16_t> dataBias0(
      arg2_bias0_Dims[0] * arg2_bias0_Dims[1], 1.f);
  std::vector<bfloat16_t> dataWeight1(
      arg3_weight1_Dims[0] * arg3_weight1_Dims[1], 1.f);
  std::vector<bfloat16_t> dataBias1(
      arg4_bias1_Dims[0] * arg4_bias1_Dims[1], 1.f);
  std::vector<bfloat16_t> expectedOutput(
      arg5_output_Dims[0] * arg5_output_Dims[1], 257.f);

  // Transfer the first tensor, dataInput, to the device.
  constexpr uint64_t inputDMemAddress = 0;
  kernelInputs.push_back(
      KernelParams::ArrayRef{arg0_input_Dims, inputDMemAddress});

  // Transfer the second tensor, dataWeight0, to the device.
  const uint64_t weight0DMemAddress =
      inputDMemAddress + dataInput.size() * sizeof(bfloat16_t);
  kernelInputs.push_back(
      KernelParams::ArrayRef{arg1_weight0_Dims, weight0DMemAddress});

  // Transfer the third tensor, dataBias0, to the device.
  const uint64_t bias0DMemAddress =
      weight0DMemAddress + dataWeight0.size() * sizeof(bfloat16_t);
  kernelInputs.push_back(
      KernelParams::ArrayRef{arg2_bias0_Dims, bias0DMemAddress});

  // Transfer the fourth tensor, dataWeight1, to the device.
  const uint64_t weight1DMemAddress =
      bias0DMemAddress + dataBias0.size() * sizeof(bfloat16_t);
  kernelInputs.push_back(
      KernelParams::ArrayRef{arg3_weight1_Dims, weight1DMemAddress});

  // Transfer the fifth tensor, dataBias1, to the device.
  const uint64_t bias1DMemAddress = bias0DMemAddress + NativeMatrixDim *
                                                           dataWeight1.size() *
                                                           sizeof(bfloat16_t);
  kernelInputs.push_back(
      KernelParams::ArrayRef{arg4_bias1_Dims, bias1DMemAddress});

  // Define result address
  const uint64_t resultDMemAddress = bias1DMemAddress + NativeMatrixDim *
                                                            dataBias1.size() *
                                                            sizeof(bfloat16_t);
  kernelOutputs.push_back(
      KernelParams::ArrayRef{arg5_output_Dims, resultDMemAddress});

  // Create ClusterCommand args
  auto args = CreateKernelParams(kernelInputs, kernelOutputs);

  // Re-Initialize DMem with host pre-allocations
  uint64_t dMemOffset =
      resultDMemAddress + expectedOutput.size() * sizeof(bfloat16_t);

  // device.template InitializeMemoryManager<bfloat16_t>(dMemOffset);
  device.InitializeMemoryManager(dMemOffset);
  uint32_t retCode = device.RetrieveMessageWord();

  // Re-Initialize the cluster semaphore stack to begin after an offset of 13
  // for testing.
  // device.template InitializeSemaphoreStack<bfloat16_t>(13);
  device.InitializeSemaphoreStack(13);
  retCode = device.RetrieveMessageWord();

  device.WriteToHbm<bfloat16_t>(
      inputDMemAddress, dataInput, Apollo::Primitives::DataType::BFloat16);
  device.WriteToHbm<bfloat16_t>(
      weight0DMemAddress, dataWeight0, Apollo::Primitives::DataType::BFloat16);
  device.WriteToHbm<bfloat16_t>(
      bias0DMemAddress, dataBias0, Apollo::Primitives::DataType::BFloat16);
  device.WriteToHbm<bfloat16_t>(
      weight1DMemAddress, dataWeight1, Apollo::Primitives::DataType::BFloat16);
  device.WriteToHbm<bfloat16_t>(
      bias1DMemAddress, dataBias1, Apollo::Primitives::DataType::BFloat16);

  constexpr size_t funcId = Trainwave::FirmwareSDK::GetCommandID<
      MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand,
      MaiaCompiler::GeneratedCode::ClusterCP::Interface>();

  Apollo::Message::ControlMessage ctrlMsg =
      Apollo::Message::WrapIntoControlMessage(args, funcId);

#ifndef NDEBUG
  std::cout << "Host: Sending command ..." << std::endl;
#endif

  device.ExecuteAsync(ctrlMsg);
  retCode = device.RetrieveMessageWord();

  assert(funcId == retCode);

#ifndef NDEBUG
  std::cout << "Host: Validating result ..." << std::endl;
#endif
  auto outBytes = device.ReadFromHbm(kernelOutputs.at(0).dramAddress,
      expectedOutput.size(), Apollo::Primitives::DataType::BFloat16);

  // Reinterpret the byte buffer as a span of SKU::BFloat16Type
  auto actualResult = ReinterpretSpan<BFloat16Type>(gsl::make_span(outBytes));

  std::vector<bfloat16_t> convertedActualResult;

  std::transform(actualResult.begin(), actualResult.end(),
      std::back_inserter(convertedActualResult),
      [](BFloat16Type elem) { return elem; });

  if (convertedActualResult != expectedOutput) {
    std::cout << "Test failed ..." << std::endl;
    return 1;
  } else {
    std::cout << "Test succeeded ..." << std::endl;
    return 0;
  }
}