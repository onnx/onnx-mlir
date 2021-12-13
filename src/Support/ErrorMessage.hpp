/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- ErrorMessage.hpp --------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//
// Return error message for given error number for opening file at given path
// If error number is not given, investigate it by opening it.
//
#define PATH_SIZE 4096
static std::string getErrorMessageforFileOpeningErrors(
    const std::string &path, int msgnum = -1, int flags = -1, int mode = -1) {
  // If errno not given investigate the error by opening the path
  if (msgnum < 0) {
    flags = (flags > 0) ? flags : (O_CREAT | O_WRONLY);
    mode = (mode >= 0) ? mode : (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    int fd = open(path.c_str(), flags, mode);
    msgnum = errno;
    close(fd);
  }
  char dir[PATH_SIZE];
  getcwd(dir, PATH_SIZE);
  std::string msg = std::string(strerror(msgnum)) + "(" +
      std::to_string(msgnum) + ") for " + path + " at " + dir;
  return msg;
}
