
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Command.h - Create exec commands -------------------------===//
//
// This file contains code LLVM-agnostic calls to create exec commands. Works in
// C and C++ environment. Has a minimum set of dependences.
//
//===----------------------------------------------------------------------===//

#ifndef COMMAND_H
#define COMMAND_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
#include <stdexcept>
#include <string>
#include <vector>
#endif

// Forward declaration of Command
struct Command;
#ifndef __cplusplus
typedef struct Command Command;
#endif

// Exceptions and error codes (negative values to distinguish from exit codes)
#define CMD_SUCCESS 0
#define CMD_ERROR_NULL_PTR -1
#define CMD_ERROR_ALLOC -2
#define CMD_ERROR_EMPTY_ARG -3
#define CMD_ERROR_PATH_TOO_LONG -4
#define CMD_ERROR_CHDIR -5
#define CMD_ERROR_FORK -6
#define CMD_ERROR_EXEC -7

#ifdef __cplusplus
// Exceptions
class CommandException : public std::runtime_error {
public:
  explicit CommandException(const std::string &msg) : std::runtime_error(msg) {}
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a new Command structure
 * @param exePath Path to the executable
 * @param verbose Verbose mode on (print exec command)
 * @return Pointer to Command structure, or NULL on failure
 */
Command *commandCreate(const char *exePath, int verbose);

/**
 * Append a single string argument to the command
 * @param cmd Pointer to Command structure
 * @param arg Argument string to append
 * @return CMD_SUCCESS on success, error code on failure
 */
int commandAppendStr(Command *cmd, const char *arg);

/**
 * Append a list of string arguments to the command
 * @param cmd Pointer to Command structure
 * @param args Array of argument strings
 * @param count Number of arguments in the array
 * @return CMD_SUCCESS on success, error code on failure
 */
int commandAppendList(Command *cmd, const char **args, size_t count);

/**
 * Reset arguments to only the executable name
 * @param cmd Pointer to Command structure
 * @return CMD_SUCCESS on success, error code on failure
 */
int commandResetArgs(Command *cmd);

/**
 * Print the Command structure to a file handle
 * @param cmd Pointer to Command structure
 * @param fp File handle (stdout, stderr, or file pointer)
 * @param wdir working directory (null ok)
 */
void commandPrint(const Command *cmd, FILE *fp, const char *wdir);

/**
 * Execute the command
 * @param cmd Pointer to Command structure
 * @param wdir Working directory (NULL or empty string for current directory)
 * @return Exit code (0-255) on success, negative error code on failure
 */
int commandExec(Command *cmd, const char *wdir);

/**
 * Free Command resources
 * @param cmd Pointer to Command structure
 */
void commandDestroy(Command *cmd);

#ifdef __cplusplus
}
#endif

// Command structure
struct Command {

#ifdef __cplusplus
  // C++ Interface.
  Command(std::string exePath, bool verbose = false);
  ~Command();
  Command &appendStr(const std::string &arg);
  Command &appendList(const std::vector<std::string> &args);
  Command &resetArgs();
  Command &print(FILE *fp = nullptr, std::string wdir = "");
  int exec(std::string wdir = "");
#endif

  char *path;
  char **args;
  size_t argsCount;
  size_t argsCapacity;
  int verbose;
};

#endif // COMMAND_H
