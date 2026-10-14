/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Command.hpp - Create exec commands -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands. This file should include
// no dependences to ONNX-MLIR / MLIR / LLVM files.
//
//===----------------------------------------------------------------------===//

#ifndef COMMAND_H
#define COMMAND_H

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace onnx_mlir {

// Exception class.
class CommandException : public std::runtime_error {
public:
  explicit CommandException(const std::string &msg) : std::runtime_error(msg) {}
};

// Command class: it enables the user to call a binary with parameters and
// execute this binary in a separate process using a fork/join command.
// Parameters are added using the 'appendStr` or `appendList` functions. The
// `exec` function will invoke the fork/join. If logging of the forked process
// standard output/error is desired, calling `redirectExecStreams` before
// executing the command will redirect these streams into a single common file.
// For debugging, the `print` command emit the command to a stream (e.g. stderr)
// without executing it.
//
// Exec function may generate CommandException on failure.
class Command {

public:
  // Constructor / Destructor.
  Command(const std::string &exePath, bool verbose = false);
  ~Command() = default;

  // Delete copy, allow move.
  Command() = delete;
  Command(const Command &) = delete;
  Command &operator=(const Command &) = delete;
  Command(Command &&) = default;
  Command &operator=(Command &&) = default;

  // Append a single string argument.
  Command &appendStr(const std::string &arg);

  // Append a list of string arguments.
  Command &appendList(const std::vector<std::string> &args);

  // Get all the arguments.
  std::vector<std::string> getArgs() { return args; }

  // Reset arguments to only executable name.
  Command &resetArgs();

  // Print command to file handle.
  Command &print(FILE *fp = nullptr, const std::string &wdir = "");

  // void redirect exec streams (Linux/MacOS support only).
  void redirectExecStreams(const std::string &stdFilename);

  // Execute command.
  int exec(const std::string &wdir = "");

private:
  // Helper to get basename from path.
  static std::string getBasename(const std::string &path);

  // Helper to print escaped string.
  static void printEscapedString(FILE *fp, const std::string &str);

  std::string path;
  std::string stdFilename;
  bool verbose;
  std::vector<std::string> args;
};

} // namespace onnx_mlir

#endif // COMMAND_H
