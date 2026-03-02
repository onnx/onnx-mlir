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
#include "src/Compiler/DriverUtils.hpp"

#include <filesystem>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace onnx_mlir;

// Constructor.
Command::Command(const std::string &exePath, bool isVerbose)
    : path(exePath), stdFilename(""), verbose(isVerbose) {
  if (path.empty()) {
    throw CommandException("Empty executable path");
  }

  // Add executable name as first argument.
  args.push_back(getBasename(path));
}

// Helper to get basename from path.
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

// Append a single string argument.
Command &Command::appendStr(const std::string &arg) {
  if (!arg.empty()) {
    args.push_back(arg);
  }
  return *this;
}

// Append a list of string arguments.
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

// Print command to file handle.
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

void Command::redirectExecStreams(const std::string &stdFilename) {
  this->stdFilename = stdFilename;
}

// Execute command - Platform-specific implementations.
int Command::exec(const std::string &wdir) {
  // Get current working directory.
  fs::path curWdir = fs::current_path();
  // Determine new working directory based on wdir.
  fs::path newWdir;
  bool requestedNewDir = !wdir.empty();
  if (requestedNewDir)
    newWdir = getAbsolutePathUsingCurrentDir(wdir);
  else
    newWdir = curWdir;

  // Change directory if requested.
  if (requestedNewDir) {
    try {
      fs::current_path(newWdir);
    } catch (const fs::filesystem_error &e) {
      throw CommandException(
          "Failed to change directory to: " + newWdir.string());
    }
  }
  // Print command if verbose.
  if (verbose) {
    print(stdout, newWdir.string());
    printf("begin exec\n");
  }

#ifdef _WIN32
  // Windows implementation using CreateProcess.

  // Build command line string.
  std::string cmdLine;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      cmdLine += " ";

    // Quote arguments with spaces.
    bool needQuotes = args[i].find(' ') != std::string::npos;
    if (needQuotes)
      cmdLine += "\"";
    cmdLine += args[i];
    if (needQuotes)
      cmdLine += "\"";
  }

  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi;

  // CreateProcess modifies the command line, so we need a writable copy.
  std::vector<char> cmdLineBuf(cmdLine.begin(), cmdLine.end());
  cmdLineBuf.push_back('\0');

  if (!CreateProcessA(path.c_str(), // Application name
          cmdLineBuf.data(),        // Command line
          NULL,                     // Process security attributes
          NULL,                     // Thread security attributes
          FALSE,                    // Inherit handles
          0,                        // Creation flags
          NULL,                     // Environment
          NULL,                     // Current directory (already changed)
          &si,                      // Startup info
          &pi)) {                   // Process information

    if (requestedNewDir)
      fs::current_path(curWdir);
    throw CommandException("CreateProcess failed");
  }

  // Wait for process to complete.
  WaitForSingleObject(pi.hProcess, INFINITE);

  // Get exit code.
  DWORD exitCode;
  GetExitCodeProcess(pi.hProcess, &exitCode);

  // Clean up handles.
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  // Restore working directory.
  if (requestedNewDir) {
    try {
      fs::current_path(curWdir);
    } catch (const fs::filesystem_error &e) {
      throw CommandException("Failed to restore directory");
    }
  }

  return static_cast<int>(exitCode);

#else
  // Unix/Linux implementation using fork/exec.

  // Prepare arguments array (NULL-terminated for execvp).
  std::vector<char *> execArgs;
  execArgs.reserve(args.size() + 1);
  for (auto &arg : args) {
    execArgs.push_back(const_cast<char *>(arg.c_str()));
  }
  execArgs.push_back(nullptr);

  // Fork and execute.
  pid_t pid = fork();
  if (pid == -1) {
    if (requestedNewDir)
      fs::current_path(curWdir);
    throw CommandException("Fork failed");
  }

  if (pid == 0) {
    // Child process.
    if (!stdFilename.empty()) {
      int fd = open(stdFilename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      dup2(fd, STDOUT_FILENO); // Redirect stdout.
      dup2(fd, STDERR_FILENO); // Redirect stderr.
      close(fd);
    }
    errno = 0;
    execvp(path.c_str(), execArgs.data());
    // execvp failed
    int err = errno;
    fprintf(stderr, "Failed in Command to execute '%s': errno=%d\n",
        path.c_str(), err);

    // Use specific onnx-mlir error codes
    switch (err) {
    case ENOENT:
      exit(onnx_mlir::CommandNotFound);
    case EACCES:
      exit(onnx_mlir::CommandNotExecutable);
    default:
      exit(onnx_mlir::CommandExecutionFailed);
    }
  }

  // Parent process.
  int status;
  waitpid(pid, &status, 0);

  if (verbose)
    printf("end exec\n");

  // Restore working directory.
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
#endif
}
