/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Command.cpp - Create exec commands -----------------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// This file contains C++ code to create exec commands. This file should include
// no dependences to ONNX-MLIR / MLIR / LLVM files.
//
//===----------------------------------------------------------------------===//

#include "src/Compiler/Command.hpp"
#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace onnx_mlir;

// Helper to get basename from path
std::string Command::getBasename(const std::string &path) {
  return fs::path(path).filename().string();
}

// Helper to print escaped string
void Command::printEscapedString(FILE *fp, const std::string &str) {
  if (!fp)
    fp = stderr;

  for (char c : str) {
    switch (c) {
    case '"':
      fprintf(fp, "\\\"");
      break;
    case '\\':
      fprintf(fp, "\\\\");
      break;
    case '\n':
      fprintf(fp, "\\n");
      break;
    case '\t':
      fprintf(fp, "\\t");
      break;
    case '\r':
      fprintf(fp, "\\r");
      break;
    default:
      fputc(c, fp);
      break;
    }
  }
}

// Constructor
Command::Command(const std::string &exePath, bool isVerbose)
    : path(exePath), verbose(isVerbose) {
  if (path.empty()) {
    throw CommandException("Empty executable path");
  }

  // Add executable name as first argument
  args.push_back(getBasename(path));
}

// Append a single string argument
Command &Command::appendStr(const std::string &arg) {
  if (!arg.empty()) {
    args.push_back(arg);
  }
  return *this;
}

// Append a list of string arguments
Command &Command::appendList(const std::vector<std::string> &argList) {
  for (const auto &arg : argList) {
    if (!arg.empty()) {
      args.push_back(arg);
    }
  }
  return *this;
}

// Reset arguments to only executable name
Command &Command::resetArgs() {
  if (args.empty()) {
    throw CommandException("No arguments to reset");
  }

  std::string exeName = args[0];
  args.clear();
  args.push_back(exeName);
  return *this;
}

// Print command to file handle
Command &Command::print(FILE *fp, const std::string &wdir) {
  if (!fp)
    fp = stderr;

  fprintf(fp, "[%s] %s:", wdir.empty() ? "" : wdir.c_str(), path.c_str());
  for (const auto &arg : args) {
    fprintf(fp, " ");
    printEscapedString(fp, arg);
  }
  fprintf(fp, "\n");
  return *this;
}

// Execute command
int Command::exec(const std::string &wdir) {
  // Get current working directory
  fs::path curWdir = fs::current_path();

  // Determine new working directory
  fs::path newWdir;
  bool requestedNewDir = !wdir.empty();

  if (requestedNewDir) {
    newWdir = fs::path(wdir);
    if (newWdir.is_relative()) {
      newWdir = curWdir / newWdir;
    }
  } else {
    newWdir = curWdir;
  }

  // Print command if verbose
  if (verbose) {
    print(stdout, newWdir.string());
  }

  // Change directory if requested
  if (requestedNewDir) {
    try {
      fs::current_path(newWdir);
    } catch (const fs::filesystem_error &e) {
      throw CommandException(
          "Failed to change directory to: " + newWdir.string());
    }
  }

  // Prepare arguments array (NULL-terminated for execvp)
  std::vector<char *> execArgs;
  execArgs.reserve(args.size() + 1);
  for (auto &arg : args) {
    execArgs.push_back(const_cast<char *>(arg.c_str()));
  }
  execArgs.push_back(nullptr);

  // Fork and execute
  pid_t pid = fork();
  if (pid == -1) {
    if (requestedNewDir)
      fs::current_path(curWdir);
    throw CommandException("Fork failed");
  }

  if (pid == 0) {
    // Child process
    execvp(path.c_str(), execArgs.data());
    exit(127); // execvp failed
  }

  // Parent process
  int status;
  waitpid(pid, &status, 0);

  // Restore working directory
  if (requestedNewDir) {
    try {
      fs::current_path(curWdir);
    } catch (const fs::filesystem_error &e) {
      throw CommandException("Failed to restore directory");
    }
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }

  throw CommandException("Command execution failed");
}
