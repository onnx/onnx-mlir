/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- Command.inc - Create exec commands -----------------------===//
//
// This file contains code LLVM-agnostic calls to create exec commands. Works in
// C and C++ environment. Has a minimum set of dependences.
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "src/Compiler/Command.h"

// Set all Command fields to nothing, so that we may safely reset on errors.
static void commandSet(Command *cmd) {
  cmd->path = NULL;
  cmd->args = NULL;
  cmd->argsCount = 0;
  cmd->argsCapacity = 0;
}

// Free Command resources.
static void commandReset(Command *cmd) {
  // Since all Command fields are set to safe values, safe to execute this.
  free(cmd->path);
  for (size_t i = 0; i < cmd->argsCount; i++) {
    free(cmd->args[i]);
  }
  free(cmd->args);
}

// Initialize all of the fields.
static int commandInit(Command *cmd, const char *exePath, int verbose) {
  // Default initializations.
  if (!cmd)
    return CMD_ERROR_NULL_PTR;
  commandSet(cmd);

  // Handle executable path.
  cmd->path = strdup(exePath);
  if (!cmd->path) {
    commandReset(cmd);
    return CMD_ERROR_ALLOC;
  }
  // Allocate args capacity.
  int initialCapacity = 16;
  cmd->args = (char **)malloc(sizeof(char *) * initialCapacity);
  if (!cmd->args) {
    commandReset(cmd);
    return CMD_ERROR_ALLOC;
  }
  cmd->argsCapacity = initialCapacity;
  cmd->argsCount = 0;
  // Add executable name as first argument
  char *pathCopy = strdup(exePath);
  if (!pathCopy) {
    commandReset(cmd);
    return CMD_ERROR_ALLOC;
  }
  char *exeName = basename(pathCopy);
  cmd->args[0] = strdup(exeName);
  free(pathCopy);
  if (!cmd->args[0]) {
    commandReset(cmd);
    return CMD_ERROR_ALLOC;
  }
  // Success with first argument.
  cmd->argsCount = 1;
  cmd->verbose = verbose;
  return CMD_SUCCESS;
}

// Initialize a new Command
Command *commandCreate(const char *exePath, int verbose) {
  Command *cmd = (Command *)malloc(sizeof(Command));
  if (!cmd)
    return NULL;
  int result = commandInit(cmd, exePath, verbose);
  if (result != CMD_SUCCESS) {
    free(cmd);
    return NULL;
  }
  return cmd;
}

// Free a command;
void commandDestroy(Command *cmd) {
  if (!cmd)
    return;
  commandReset(cmd);
  free(cmd);
}

// Append a single string argument
int commandAppendStr(Command *cmd, const char *arg) {
  if (!cmd || !arg || strlen(arg) == 0)
    return CMD_SUCCESS;
  // Grow buffer if needed.
  if (cmd->argsCount >= cmd->argsCapacity) {
    cmd->argsCapacity *= 2;
    char **newArgs =
        (char **)realloc(cmd->args, sizeof(char *) * cmd->argsCapacity);
    if (!newArgs) {
      return CMD_ERROR_ALLOC;
    }
    cmd->args = newArgs;
  }
  // Copy string.
  char *argCopy = strdup(arg);
  if (!argCopy) {
    return CMD_ERROR_ALLOC;
  }
  // Add string to buffer.
  cmd->args[cmd->argsCount++] = argCopy;
  return CMD_SUCCESS;
}

// Append a list of string arguments
int commandAppendList(Command *cmd, const char **args, size_t count) {
  if (!cmd || !args)
    return CMD_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    int result = commandAppendStr(cmd, args[i]);
    if (result != CMD_SUCCESS) {
      return result;
    }
  }
  return CMD_SUCCESS;
}

// Reset arguments
int commandResetArgs(Command *cmd) {
  if (!cmd)
    return CMD_SUCCESS;
  if (cmd->argsCapacity < 1)
    return CMD_ERROR_EMPTY_ARG;
  // Reset exec name.
  char *exeName = cmd->args[0];
  // Free all arguments except the first
  for (size_t i = 1; i < cmd->argsCount; i++) {
    free(cmd->args[i]);
  }
  // Reinit.
  cmd->argsCount = 1;
  cmd->args[0] = exeName;
  return CMD_SUCCESS;
}

// Helper function to print a string with proper escaping for quotes
static void printEscapedString(FILE *fp, const char *str) {
  if (!fp)
    fp = stderr;
  // Empty command
  if (!str) {
    fprintf(fp, "(null)");
    return;
  }
  // Handle special chars while printing the string.
  for (const char *p = str; *p != '\0'; p++) {
    if (*p == '"') {
      fprintf(fp, "\\\"");
    } else if (*p == '\\') {
      fprintf(fp, "\\\\");
    } else if (*p == '\n') {
      fprintf(fp, "\\n");
    } else if (*p == '\t') {
      fprintf(fp, "\\t");
    } else if (*p == '\r') {
      fprintf(fp, "\\r");
    } else {
      fputc(*p, fp);
    }
  }
}

// Print the Command structure to a file handle
void commandPrint(const Command *cmd, FILE *fp, const char *wdir) {
  if (!fp)
    fp = stderr;
  if (!cmd) {
    fprintf(fp, "Command: (null)\n");
    return;
  }
  fprintf(fp, "[%s] %s:", (wdir ? wdir : ""), cmd->path);
  for (size_t i = 0; i < cmd->argsCount; i++) {
    fprintf(fp, " ");
    printEscapedString(fp, cmd->args[i]);
  }
  fprintf(fp, "\n");
}

// Execute command
// Returns: exit code of child process (0-255) on success, negative error code
// on failure
int commandExec(Command *cmd, const char *wdir) {
  if (!cmd)
    return CMD_ERROR_NULL_PTR;

  // Get current working directory
  char curWdir[PATH_MAX];
  if (!getcwd(curWdir, sizeof(curWdir))) {
    return CMD_ERROR_CHDIR;
  }
  // Set new working directory as specified.
  char newWdir[PATH_MAX];
  int requestedNewDir = wdir && strlen(wdir) > 0;
  if (requestedNewDir) {
    // Gave a working dir, use it.
    if (wdir[0] == '/') {
      // Absolute path - check if it fits in buffer
      size_t resultLen = strlen(wdir);
      if (resultLen >= PATH_MAX) {
        return CMD_ERROR_PATH_TOO_LONG;
      }
      strncpy(newWdir, wdir, PATH_MAX - 1);
      newWdir[PATH_MAX - 1] = '\0';
    } else {
      // Relative path - combined current dir and provided dir.
      size_t resultLen = strlen(curWdir) + 1 + strlen(wdir); // +1 for '/'
      if (resultLen >= PATH_MAX) {
        return CMD_ERROR_PATH_TOO_LONG;
      }
      int written = snprintf(newWdir, PATH_MAX, "%s/%s", curWdir, wdir);
      if (written < 0 || written >= PATH_MAX) {
        return CMD_ERROR_PATH_TOO_LONG;
      }
    }
  } else {
    // Was not given a wdir, use current dir.
    strncpy(newWdir, curWdir, PATH_MAX);
  }

  // Prepare arguments array (NULL-terminated)
  char **execArgs = (char **)malloc(sizeof(char *) * (cmd->argsCount + 1));
  if (!execArgs) {
    return CMD_ERROR_ALLOC;
  }
  for (size_t i = 0; i < cmd->argsCount; i++) {
    execArgs[i] = cmd->args[i];
  }
  execArgs[cmd->argsCount] = NULL;

  // Fork and execute
  if (cmd->verbose)
    commandPrint(cmd, stdout, newWdir);

  // Change directory.
  if (requestedNewDir && chdir(newWdir) != 0) {
    free(execArgs);
    return CMD_ERROR_CHDIR;
  }
  pid_t pid = fork();
  if (pid == -1) {
    free(execArgs);
    if (requestedNewDir)
      chdir(curWdir);
    return CMD_ERROR_FORK;
  }

  if (pid == 0) {
    // Child process
    execvp(cmd->path, execArgs);
    exit(127); // execvp failed
  }

  // Parent process
  int status;
  waitpid(pid, &status, 0);
  free(execArgs);

  // Restore working directory
  if (requestedNewDir && chdir(curWdir) != 0) {
    return CMD_ERROR_CHDIR;
  }

  if (WIFEXITED(status)) {
    // Return the actual exit code from the child process
    return WEXITSTATUS(status);
  }
  return CMD_ERROR_EXEC;
}

#ifdef __cplusplus

Command::Command(std::string exePath, bool isVerbose)
    // Same init as in commandSet().
    : path(nullptr), args(nullptr), argsCount(0), argsCapacity(0) {
  const char *exePathPtr = exePath.empty() ? nullptr : exePath.c_str();
  int result = commandInit(this, exePathPtr, isVerbose ? 1 : 0);
  if (result != CMD_SUCCESS)
    throw CommandException("Failed to initialize command");
}

Command::~Command() { commandReset(this); }

Command &Command::appendStr(const std::string &arg) {
  const char *argPtr = arg.empty() ? nullptr : arg.c_str();
  int result = commandAppendStr(this, argPtr);
  if (result == CMD_ERROR_ALLOC) {
    throw CommandException("Memory allocation failed in appendStr");
  }
  return *this;
}

Command &Command::appendList(const std::vector<std::string> &args) {
  for (const auto &arg : args) {
    appendStr(arg);
  }
  return *this;
}

Command &Command::resetArgs() {
  int result = commandResetArgs(this);
  if (result == CMD_ERROR_EMPTY_ARG)
    throw CommandException("Failed to reset command");
  return *this;
}

Command &Command::print(FILE *fp, std::string wdir) {
  const char *wdirPtr = wdir.empty() ? nullptr : wdir.c_str();
  commandPrint(this, fp, wdirPtr);
  return *this;
}

int Command::exec(std::string wdir) {
  const char *wdirPtr = wdir.empty() ? nullptr : wdir.c_str();
  int result = commandExec(this, wdirPtr);
  if (result == CMD_ERROR_ALLOC) {
    throw CommandException("Memory allocation failed in exec");
  } else if (result == CMD_ERROR_PATH_TOO_LONG) {
    throw CommandException("Path too long in exec");
  } else if (result == CMD_ERROR_CHDIR) {
    throw CommandException("Failed to change directory in exec");
  } else if (result == CMD_ERROR_FORK) {
    throw CommandException("Fork failed in exec");
  } else if (result == CMD_ERROR_EXEC) {
    throw CommandException("Execution failed in exec");
  } else if (result < 0) {
    throw CommandException("Unknown error in exec");
  }

  // Return the exit code (0-255)
  return result;
}

#endif // __cplusplus
