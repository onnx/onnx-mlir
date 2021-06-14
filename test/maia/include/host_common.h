
#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tiling.h"

using npufloat_t = SKU::BFloat16Type;
constexpr auto TILE_SIZE = SKU::NativeMatrixDim;

namespace fs = std::filesystem;
namespace twr = Trainwave::Runtime;
namespace twe = Trainwave::Emulator;

template <typename T>
using span = gsl::span<T>;

template <typename T, typename U>
span<T> AsSpan(U &obj) {
  static_assert(sizeof(U) % sizeof(T) == 0);
  return span<T>(reinterpret_cast<T *>(&obj), sizeof(U) / sizeof(T));
}

inline MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments
CreateKernelParams(std::vector<KernelParams::ArrayRef> &kernelInputs,
    std::vector<KernelParams::ArrayRef> &kernelOutputs) {
  MaiaCompiler::GeneratedCode::ClusterCP::ClusterCommand::Arguments args{};
  args.params.numInputTensors = gsl::narrow<uint8_t>(kernelInputs.size());
  args.params.numOutputTensors = gsl::narrow<uint8_t>(kernelOutputs.size());
  span<uint32_t> buffer(args.params.data);

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

inline size_t compute_size(KernelParams::ArrayRef::DimsType &dims) {
  size_t size = 1;
  for (auto dim : dims)
    size *= std::max<size_t>(dim, 1);
  return size;
}

enum class InOut { Input, Output };

std::vector<npufloat_t> load_data(
    fs::path &data_loc, InOut source, std::string_view name) {
  auto path = data_loc / (source == InOut::Input ? "inputs" : "outputs") / name;

  try {
    std::ifstream data_file;
    data_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    data_file.open(path, std::ios::binary);

    std::vector<unsigned char> data_raw(
        std::istreambuf_iterator<char>(data_file), {});
    auto data_as_span = ReinterpretSpan<float>(gsl::make_span(data_raw));
    std::vector<npufloat_t> data;
    std::transform(data_as_span.begin(), data_as_span.end(),
        std::back_insert_iterator(data), [](npufloat_t elem) { return elem; });
    return data;
  } catch (std::exception &e) {
    std::cerr << "failed to load '" << name << "' from " << path << std::endl;
    throw;
  }
}
