#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <boost/program_options.hpp>

#include "src/builder/sgir.hpp"

#include "mlir/IR/Module.h"

using namespace std;
using namespace onnf;

int main(int ac, char* av[]) {
  namespace po = boost::program_options;

  po::options_description desc("ONNF available options");
  // clang-format off
  desc.add_options()("help", "produce help message")(
      "onnx-model", po::value<string>()->required(),
        "onnx model file");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  string model_filename = vm["onnx-model"].as<string>();
  auto module = SGIRImportModelFile(model_filename);

  return 0;
}
