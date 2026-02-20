/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Command.hpp - Create exec commands -----------------------===//
//
// This file contains C++ code to create exec commands.
//
//===----------------------------------------------------------------------===//

#ifndef COMMAND_H
#define COMMAND_H

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Exception class
class CommandException : public std::runtime_error {
public:
  explicit CommandException(const std::string &msg) : std::runtime_error(msg) {}
};

// Command class
class Command {
public:
  std::string path;
  std::vector<std::string> args;
  bool verbose;

private:
  // Helper to get basename from path
  static std::string getBasename(const std::string &path);

  // Helper to print escaped string
  static void printEscapedString(FILE *fp, const std::string &str);

public:
  // Constructor
  Command(const std::string &exePath, bool verbose = false);

  // Destructor (default is fine)
  ~Command() = default;

  // Delete copy, allow move
  Command(const Command &) = delete;
  Command &operator=(const Command &) = delete;
  Command(Command &&) = default;
  Command &operator=(Command &&) = default;

  // Append a single string argument
  Command &appendStr(const std::string &arg);

  // Append a list of string arguments
  Command &appendList(const std::vector<std::string> &args);

  // Get all the arguments.
  std::vector<std::string> getArgs() { return args; }

  // Reset arguments to only executable name
  Command &resetArgs();

  // Print command to file handle
  Command &print(FILE *fp = nullptr, const std::string &wdir = "");

  // Execute command
  int exec(const std::string &wdir = "");
};

#endif // COMMAND_H
