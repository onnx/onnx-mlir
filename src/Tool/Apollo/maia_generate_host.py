import argparse
import sys

import onnxruntime


def emit_includes():
  out_file.write('#include <cstddef>\n')
  out_file.write('#include <iomanip>\n')
  out_file.write('#include <iostream>\n')
  out_file.write('#include <fstream>\n')
  out_file.write('#include <filesystem>\n')
  out_file.write('#include <vector>\n')
  out_file.write('#include <string>\n')
  out_file.write('#include <string_view>\n')
  out_file.write('#include <limits>\n')
  out_file.write('#include <utility>\n')
  out_file.write('\n')
  out_file.write('#include "Commands/CommandMappingUtils.h"\n')
  out_file.write('#include "ControlMessage.h"\n')
  out_file.write('#include "KernelParams.h"\n')
  out_file.write('#include "Runtime/Device.h"\n')
  out_file.write('\n')


def emit_helpers():
  out_file.write('namespace fs = std::filesystem;\n')
  out_file.write('namespace twr = Trainwave::Runtime;\n')
  out_file.write('namespace twe = Trainwave::Emulator;\n')
  out_file.write('using SKU = SkuTraits<Sku_NpuTransformersSP>;\n')
  out_file.write('using bfloat_t = SKU::BFloat16Type;\n')
  out_file.write('template<typename T>\n')
  out_file.write('using span = gsl::span<T>;\n')
  out_file.write('\n')
  out_file.write('template <typename T, typename U>\n')
  out_file.write('span<T> AsSpan(U &obj) {\n')
  out_file.write('  static_assert(sizeof(U) % sizeof(T) == 0);\n')
  out_file.write('  return span<T>(reinterpret_cast<T *>(&obj), sizeof(U) / sizeof(T));\n')
  out_file.write('}\n')
  out_file.write('\n')
  out_file.write('MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments\n')
  out_file.write('CreateKernelParams(std::vector<KernelParams::ArrayRef> &kernelInputs,\n')
  out_file.write('    std::vector<KernelParams::ArrayRef> &kernelOutputs) {\n')
  out_file.write('  MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments args{};\n')
  out_file.write('  args.params.numInputTensors = gsl::narrow<uint8_t>(kernelInputs.size());\n')
  out_file.write('  args.params.numOutputTensors = gsl::narrow<uint8_t>(kernelOutputs.size());\n')
  out_file.write('  span<uint32_t> buffer(args.params.data);\n')
  out_file.write('\n')
  out_file.write('  auto it = buffer.begin();\n')
  out_file.write('  for (auto const &ref : kernelInputs) {\n')
  out_file.write('    auto src = AsSpan<const uint32_t>(ref);\n')
  out_file.write('    it = std::copy(src.begin(), src.end(), it);\n')
  out_file.write('  }\n')
  out_file.write('  for (auto const &ref : kernelOutputs) {\n')
  out_file.write('    auto src = AsSpan<const uint32_t>(ref);\n')
  out_file.write('    it = std::copy(src.begin(), src.end(), it);\n')
  out_file.write('  }\n')
  out_file.write('  return args;\n')
  out_file.write('}\n')
  out_file.write('\n')
  out_file.write('size_t compute_size(KernelParams::ArrayRef::DimsType &dims) {\n')
  out_file.write('    size_t size = 1;\n')
  out_file.write('    for (auto dim : dims)\n')
  out_file.write('        size *= std::max<size_t>(dim, 1);\n')
  out_file.write('    return size;\n')
  out_file.write('}\n')
  out_file.write('\n')


def emit_execute_model():
  out_file.write('void ExecuteModel(std::string_view package')
  for index, input in enumerate(session.get_inputs()):
    out_file.write(f', span<bfloat_t const> arg{index}_data')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f', span<bfloat_t> out{index}_data')
  out_file.write(')\n{\n')

  for index, input in enumerate(session.get_inputs()):
    shape = str(input.shape).replace('[', '{').replace(']', '}')
    out_file.write(f'  KernelParams::ArrayRef::DimsType arg{index}_dims = {shape};\n')
  for index, output in enumerate(session.get_outputs()):
    shape = str(output.shape).replace('[', '{').replace(']', '}')
    out_file.write(f'  KernelParams::ArrayRef::DimsType out{index}_dims = {shape};\n')
  out_file.write('\n')

  for index, input in enumerate(session.get_inputs()):
    out_file.write(f'  assert(arg{index}_data.size() == compute_size(arg{index}_dims));\n')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'  assert(out{index}_data.size() == compute_size(out{index}_dims));\n')
  out_file.write('\n')

  out_file.write('  const uint64_t arg0_address = Apollo::HbmAllocations::HbmRuntimeReservedEnd;\n')
  prev_var = 'arg0'
  for index, input in enumerate(session.get_inputs()):
    if index == 0:
      continue
    out_file.write(f'  const uint64_t arg{index}_address = {prev_var}_address + {prev_var}_data.size() * sizeof(bfloat_t);\n')
    prev_var = f'arg{index}'
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'  const uint64_t out{index}_address = {prev_var}_address + {prev_var}_data.size() * sizeof(bfloat_t);\n')
    prev_var = f'out{index}'
  out_file.write('\n')

  out_file.write('  std::vector<KernelParams::ArrayRef> kernelInputs;\n')
  out_file.write('  std::vector<KernelParams::ArrayRef> kernelOutputs;\n')
  for index, input in enumerate(session.get_inputs()):
    out_file.write(f'  kernelInputs.push_back(KernelParams::ArrayRef{{arg{index}_dims, arg{index}_address}});\n')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'  kernelOutputs.push_back(KernelParams::ArrayRef{{out{index}_dims, out{index}_address}});\n')
  out_file.write('  auto args = CreateKernelParams(kernelInputs, kernelOutputs);\n')
  out_file.write('\n')

  out_file.write('  // Load the NPU Program.\n')
  out_file.write('  auto options = Trainwave::DeviceOptions();\n')
  out_file.write('  options.skuName = SKU::Name;\n')
  out_file.write('  auto device = twr::Device::CreateEmulatorDevice(options);\n')
  out_file.write('  device.LoadNPUProgram(package);\n')
  out_file.write('\n')

  out_file.write('  // send the data to the device\n')
  for index, input in enumerate(session.get_inputs()):
    out_file.write(f'  device.WriteToHbm<bfloat_t>(arg{index}_address, arg{index}_data, Apollo::Primitives::DataType::BFloat16);\n')
  out_file.write('\n')

  out_file.write('  // invoke the kernel\n')
  out_file.write('  constexpr size_t funcId = Trainwave::FirmwareSDK::GetCommandID<MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand, MaiaCompiler::GeneratedCode::ClusterCP::Interface>();\n')
  out_file.write('  Apollo::Message::ControlMessage ctrlMsg = Apollo::Message::WrapIntoControlMessage(args, funcId);\n')
  out_file.write('  device.ExecuteAsync(ctrlMsg);\n')
  out_file.write('  auto retCode = device.RetrieveMessageWord();\n')
  out_file.write('  assert(funcId == retCode);\n')
  out_file.write('\n')

  out_file.write('  // read the results\n')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'  auto out{index}_rawdata = device.ReadFromHbm(out{index}_address, compute_size(out{index}_dims), Apollo::Primitives::DataType::BFloat16);\n')
    out_file.write(f'  auto out{index}_span = ReinterpretSpan<bfloat_t>(gsl::make_span(out{index}_rawdata));\n')
    out_file.write(f'  std::transform(out{index}_span.begin(), out{index}_span.end(), out{index}_data.begin(), [](bfloat_t elem) {{ return elem; }});\n')

  out_file.write('}\n')
  out_file.write('\n')


def emit_load_data():
  out_file.write('enum class InOut { Input, Output };\n')
  out_file.write('\n')
  out_file.write('std::vector<bfloat_t> load_data(fs::path &data_loc, InOut source, std::string_view name) {\n')
  out_file.write('  auto path = data_loc / (source == InOut::Input ? "inputs" : "outputs") / name;\n')
  out_file.write('\n')
  out_file.write('  try {\n')
  out_file.write('    std::ifstream data_file;\n')
  out_file.write('    data_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);\n')
  out_file.write('    data_file.open(path, std::ios::binary);\n')
  out_file.write('\n')
  out_file.write('    std::vector<unsigned char> data_raw(std::istreambuf_iterator<char>(data_file), {});\n')
  out_file.write('    auto data_as_span = ReinterpretSpan<bfloat_t>(gsl::make_span(data_raw));\n')
  out_file.write('    std::vector<bfloat_t> data;\n')
  out_file.write('    std::transform(data_as_span.begin(), data_as_span.end(), std::back_insert_iterator(data), [](bfloat_t elem) { return elem; });\n')
  out_file.write('    return data;\n')
  out_file.write('  } catch (std::exception &e) {\n')
  out_file.write('    std::cerr << "failed to load \'" << name << "\' from " << path << std::endl;\n')
  out_file.write('    throw;\n')
  out_file.write('  }\n')
  out_file.write('}\n')
  out_file.write('\n')


def emit_main():
  out_file.write('int main(int argc, char *argv[])\n')
  out_file.write('{\n')
  out_file.write('  try {\n')
  out_file.write('    auto data_loc = fs::path{argv[0]}.parent_path().parent_path();\n')
  out_file.write('\n')
  for index, input in enumerate(session.get_inputs()):
    out_file.write(f'    auto arg{index} = load_data(data_loc, InOut::Input, "{input.name}");\n')
  out_file.write('\n')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'    auto out{index}_expected = load_data(data_loc, InOut::Output, "{output.name}");\n')
    out_file.write(f'    auto out{index} = std::vector<bfloat_t>(out{index}_expected.size());\n')
  out_file.write('\n')
  out_file.write('    // row major -> tiled format code would go here\n')
  out_file.write('\n')
  out_file.write('    ExecuteModel(argv[1]')
  for index, input in enumerate(session.get_inputs()):
    out_file.write(f', arg{index}')
  for index, output in enumerate(session.get_outputs()):
    out_file.write(f', out{index}')
  out_file.write(');\n')
  out_file.write('\n')

  for index, output in enumerate(session.get_outputs()):
    out_file.write(f'    if (out{index} != out{index}_expected) {{\n')
    out_file.write(f'      throw std::exception("\'{output.name}\' did not match expected output");\n')
    out_file.write('    }\n')
    out_file.write('\n')
  out_file.write('  } catch (std::exception &e) {\n')
  out_file.write('    std::cerr << "FAILED:" << e.what() << std::endl;\n')
  out_file.write('    return 1;\n')
  out_file.write('  }\n')
  out_file.write('\n')
  out_file.write('  std::cout << "Test succeeded." << std::endl;\n')
  out_file.write('  return 0;\n')
  out_file.write('}\n')

parser = argparse.ArgumentParser(description='Execute an ONNX test case on CPU')
parser.add_argument('--input', dest='graph_path', action='store', default="",
                    help='path-to-input-graph (ONNX file format)', required=True)
parser.add_argument('--output', dest='out_path', action='store', default="",
                    help='path-to-output-file (defaults to stdout)')
parser.add_argument('--interface-header', dest='interface_header', action='store', default="tcp_driver.dcp.h",
                    help='header for DCP inteface (defaults to: tcp_driver.dcp.h)')
args = parser.parse_args()

graph_path = args.graph_path
out_path = args.out_path
interface_header = args.interface_header

out_file = sys.stdout
if out_path != "":
  out_file = open(out_path, 'w')

session = onnxruntime.InferenceSession(graph_path)

session.get_modelmeta()

out_file.write(f'#include "{interface_header}"\n')
out_file.write('\n')
emit_includes()
emit_helpers()
emit_execute_model()
emit_load_data()
emit_main()
