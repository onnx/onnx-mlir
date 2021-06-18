import argparse
import sys

import onnxruntime

DefaultInterfaceHeader = 'tcp_driver.dcp.h'


# Format an array as a C++ array initializer
#   [a, b, c] => '{a, b, c}'
def format_shape(shape) -> str:
  return str(shape).replace('[', '{').replace(']', '}')


def generate_host(out_path: str, interface_header: str, sku: str, inputs, outputs, firstPort: int = 0):
  out_file = open(out_path, 'w') if out_path != '' else sys.stdout
  interface_header = DefaultInterfaceHeader if interface_header == '' else interface_header

  # emit includes
  out_file.write(f'#include "{interface_header}"\n')
  out_file.write('\n')
  out_file.write('#include "Commands/CommandMappingUtils.h"\n')
  out_file.write('#include "ControlMessage.h"\n')
  out_file.write('#include "KernelParams.h"\n')
  out_file.write('#include "Runtime/Device.h"\n')
  out_file.write('\n')
  out_file.write(f'using SKU = SkuTraits<{sku}>;\n')
  out_file.write('\n')
  out_file.write('#include "host_common.h"\n')
  out_file.write('\n')

  # emit "ExecuteModel" function
  out_file.write('void ExecuteModel(std::string_view package')
  for index, input in enumerate(inputs):
    out_file.write(f', span<npufloat_t const> arg{index}_data')
  for index, output in enumerate(outputs):
    out_file.write(f', span<npufloat_t> out{index}_data')
  out_file.write(')\n{\n')

  out_file.write('  // Load the NPU Program.\n')
  out_file.write('  auto options = Trainwave::DeviceOptions();\n')
  out_file.write('  options.skuName = SKU::Name;\n')
  if firstPort != 0:
    out_file.write(f'  SetNetworkConfiguration(options, {firstPort});\n')
  out_file.write('  auto device = twr::Device::CreateEmulatorDevice(options);\n')
  out_file.write('  device.LoadNPUProgram(package);\n')
  out_file.write('\n')

  for index, input in enumerate(inputs):
    out_file.write(f'  KernelParams::ArrayRef::DimsType arg{index}_dims = {format_shape(input.shape)};\n')
  for index, output in enumerate(outputs):
    out_file.write(f'  KernelParams::ArrayRef::DimsType out{index}_dims = {format_shape(output.shape)};\n')
  out_file.write('\n')

  for index, input in enumerate(inputs):
    out_file.write(f'  assert(arg{index}_data.size() == compute_size(arg{index}_dims));\n')
  for index, output in enumerate(outputs):
    out_file.write(f'  assert(out{index}_data.size() == compute_size(out{index}_dims));\n')
  out_file.write('\n')

  for index, input in enumerate(inputs):
    out_file.write(f'  const uint64_t arg{index}_address = device.AllocMemory(arg{index}_data.size() * sizeof(npufloat_t));\n')
  for index, output in enumerate(outputs):
    out_file.write(f'  const uint64_t out{index}_address = device.AllocMemory(out{index}_data.size() * sizeof(npufloat_t));\n')
  out_file.write('\n')

  out_file.write('  std::vector<KernelParams::ArrayRef> kernelInputs;\n')
  out_file.write('  std::vector<KernelParams::ArrayRef> kernelOutputs;\n')
  for index, input in enumerate(inputs):
    out_file.write(f'  kernelInputs.push_back(KernelParams::ArrayRef{{arg{index}_dims, arg{index}_address}});\n')
  for index, output in enumerate(outputs):
    out_file.write(f'  kernelOutputs.push_back(KernelParams::ArrayRef{{out{index}_dims, out{index}_address}});\n')
  out_file.write('  auto args = CreateKernelParams(kernelInputs, kernelOutputs);\n')
  out_file.write('\n')

  out_file.write('  // send the data to the device\n')
  for index, input in enumerate(inputs):
    out_file.write(f'  int16_t arg{index}_wait = {900 + index};\n')
    out_file.write(f'  device.WriteToHbm<npufloat_t>(arg{index}_address, arg{index}_data, Apollo::Primitives::DataType::BFloat16, -1, arg{index}_wait);\n')
  out_file.write('\n')

  out_file.write('  // WORKAROUND: not yet passing the argument semaphores to device, so wait here to ensure copy completed\n')
  for index, input in enumerate(inputs):
    out_file.write(f'  device.ReadFromHbm(arg{index}_address, compute_size(arg{index}_dims), Apollo::Primitives::DataType::BFloat16, arg{index}_wait, -1);\n')
  out_file.write('\n')

  out_file.write('  // invoke the kernel\n')
  out_file.write('  constexpr size_t funcId = Trainwave::FirmwareSDK::GetCommandID<MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand, MaiaCompiler::GeneratedCode::ClusterCP::Interface>();\n')
  out_file.write('  Apollo::Message::ControlMessage ctrlMsg = Apollo::Message::WrapIntoControlMessage(args, funcId);\n')
  out_file.write('  device.ExecuteAsync(ctrlMsg);\n')
  out_file.write('  auto retCode = device.RetrieveMessageWord();\n')
  out_file.write('  assert(funcId == retCode);\n')
  out_file.write('\n')

  out_file.write('  // read the results\n')
  for index, output in enumerate(outputs):
    out_file.write(f'  auto out{index}_rawdata = device.ReadFromHbm(out{index}_address, compute_size(out{index}_dims), Apollo::Primitives::DataType::BFloat16);\n')
    out_file.write(f'  auto out{index}_span = ReinterpretSpan<npufloat_t>(gsl::make_span(out{index}_rawdata));\n')
    out_file.write(f'  std::transform(out{index}_span.begin(), out{index}_span.end(), out{index}_data.begin(), [](npufloat_t elem) {{ return elem; }});\n')

  out_file.write('}\n')
  out_file.write('\n')

  # emit main() function
  out_file.write('int main(int argc, char *argv[])\n')
  out_file.write('{\n')
  out_file.write('  try {\n')
  out_file.write('    auto data_loc = fs::path{argv[0]}.parent_path();\n')
  out_file.write('\n')
  for index, input in enumerate(inputs):
    out_file.write(f'    auto arg{index} = load_data(data_loc, InOut::Input, "{input.name}");\n')
  out_file.write('\n')
  for index, output in enumerate(outputs):
    out_file.write(f'    auto out{index}_expected = load_data(data_loc, InOut::Output, "{output.name}");\n')
    out_file.write(f'    auto out{index} = std::vector<npufloat_t>(out{index}_expected.size());\n')
  out_file.write('\n')
  out_file.write('    // row major -> tiled format\n')
  for index, input in enumerate(inputs):
    out_file.write(f'    auto arg{index}_shape = std::vector<size_t>{format_shape(input.shape)};\n')
    out_file.write(f'    arg{index} = RowMajorToTileMajor<npufloat_t, TILE_SIZE>(arg{index}, arg{index}_shape);\n')
  out_file.write('\n')
  out_file.write('    ExecuteModel(argv[1]')
  for index, input in enumerate(inputs):
    out_file.write(f', arg{index}')
  for index, output in enumerate(outputs):
    out_file.write(f', out{index}')
  out_file.write(');\n')
  out_file.write('\n')
  out_file.write('    // tiled format -> row major\n')
  for index, output in enumerate(outputs):
    out_file.write(f'    auto out{index}_shape = std::vector<size_t>{format_shape(output.shape)};\n')
    out_file.write(f'    out{index} = TileMajorToRowMajor<npufloat_t, TILE_SIZE>(out{index}, out{index}_shape);\n')
  out_file.write('\n')
  for index, output in enumerate(outputs):
    out_file.write(f'    if (out{index} != out{index}_expected) {{\n')  
    out_file.write(f'      std::cout << "value  |  expected  |  actual" << std::endl;\n')
    out_file.write(f'      int failcount = 0;\n')
    out_file.write(f'      for (int i = 0; i < out{index}.size(); ++i) {{\n')
    out_file.write(f'        if (out{index}[i] != out{index}_expected[i]) {{\n')
    out_file.write(f'          if (++failcount >= 100) {{\n')
    out_file.write(f'            std::cout << "only showing first 100 errors" << std::endl;\n')
    out_file.write(f'            break;\n')
    out_file.write(f'          }}\n')
    out_file.write(f'          std::cout << i << "  |  " << out{index}_expected[i] << "  |  " << out{index}[i] << std::endl;\n')
    out_file.write(f'        }}\n')
    out_file.write(f'      }}\n')
    out_file.write('\n')
    out_file.write(f'      throw std::logic_error("\'{output.name}\' did not match expected output");\n')
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Execute an ONNX test case on CPU')
  parser.add_argument('--input', dest='graph_path', action='store', default="",
                      help='path-to-input-graph (ONNX file format)', required=True)
  parser.add_argument('--output', dest='out_path', action='store', default="",
                      help='path-to-output-file (defaults to stdout)')
  parser.add_argument('--interface-header', dest='interface_header', action='store', default=DefaultInterfaceHeader,
                      help=f'header for DCP inteface (defaults to: {DefaultInterfaceHeader})')
  parser.add_argument('--sku', dest='sku', action='store', default="Sku_NpuTransformersSP",
                      help='the target SKU (defaults to: Sku_NpuTransformersSP). Valid options: Sku_NpuTransformersSP", "Sku_ArrayFloat32"')
  parser.add_argument('--first-port', dest='first_port', action='store', default="0",
                      help='override the port range used by the device to [first_port, first_port + 4]')
  args = parser.parse_args()

  # load the graph to parse inputs/outputs
  session = onnxruntime.InferenceSession(args.graph_path)
  session.get_modelmeta()
  inputs = session.get_inputs()
  outputs = session.get_outputs()

  generate_host(args.out_path, args.interface_header, args.sku, inputs, outputs, args.first_port)

