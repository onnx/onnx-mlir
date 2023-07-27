/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OnnxMlirCompiler.h - ONNX-MLIR Compiler API Declarations -----===//
//
// This file contains declaration of onnx-mlir compiler functionality
// exported from the OnnxMlirCompiler library
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ONNXMLIRCOMPILER_H
#define ONNX_MLIR_ONNXMLIRCOMPILER_H

#include <onnx-mlir/Compiler/OMCompilerTypes.h>
#include <string>
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif // #ifdef __cplusplus

#ifdef ONNX_MLIR_BUILT_AS_STATIC
#define ONNX_MLIR_EXPORT
#else
#ifdef _MSC_VER
#ifdef OnnxMlirCompiler_EXPORTS
/* We are building this library */
#define ONNX_MLIR_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define ONNX_MLIR_EXPORT __declspec(dllimport)
#endif
#else
#define ONNX_MLIR_EXPORT __attribute__((__visibility__("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
namespace onnx_mlir {
#endif

/*!
 *  C interface to compile an onnx model from a file via onnx-mlir command.
 *  This interface is thread safe, and does not take any flags from the
 *  current environment. All flags are passed by using the flags parameter,
 *  including the "-o output-file-name" option or the "-EmitXXX" options. All
 *  options that are available to onnx-mlir are also available here.
 *
 *  This call rely on executing onnx-mlir compiler. The user can override its
 *  default location by using the ONNX_MLIR_BIN_PATH environment variable.
 *
 *  When generating libraries or jar files, the compiler will link in
 *  lightweight runtimes / jar files. If these libraries / jar files are not in
 *  the system wide directory (typically /usr/local/lib), the user can override
 *  the default location using the ONNX_MLIR_LIBRARY_PATH environment variable.
 *
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  Name may include a path, and must include the file name and its extention.
 *
 *  @param outputFilename Output file name of the compiled output for the given
 *  emission target. User is responsible for freeing the string.
 *
 *  @param flags A char * contains all the options provided to compile the
 *  model.
 *
 *  @param errorMessage Output error message, if any. User is responsible for
 *  freeing the string.
 *
 *  @return 0 on success or OnnxMlirCompilerErrorCodes on failure.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *flags, char **outputFilename, char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array. This method is not thread
 *  safe, and borrows the current compiler options currently defined in this
 *  process. When generating libraries or jar files, the compiler will link in
 *  lightweight runtimes / jar files. If these libraries / jar files are not in
 *  the system wide directory (typically /usr/local/lib), the user can override
 *  the default location using the ONNX_MLIR_LIBRARY_PATH environment variable.
 *
 *  @param inputBuffer ONNX protobuf array.
 *  @param bufferSize Size of ONNX protobuf array.
 *  @param outputBaseName File name without extension to write output.
 *  Name may include a path, must include the file name, and should not include
 *  an extention.
 *  @param emissionTarget Target format to compile to.
 *  @param outputFilename Output file name of the compiled output for the given
 *  emission target. User is responsible for freeing the string.
 *  @param errorMessage Error message, if any. User is responsible for freeing
 *  the string.
 *  @return 0 on success or OnnxMlirCompilerErrorCodes failure. User is
 *  responsible for freeing the string.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int64_t bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget, char **outputFilename,
    char **errorMessage);

/*!
 * Compute the file name of the compiled output for the given
 * emission target. User is responsible for freeing the string.
 *
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  Name may include a path, and must include the file name and its extention.
 *  @param flags A char * contains all the options provided to compile the
 *  model.
 *  @return string containing the file name. User is responsible for freeing the
 *  string.
 */
ONNX_MLIR_EXPORT char *omCompileOutputFileName(
    const char *inputFilename, const char *flags);

/*!
 * Compute the model tag from the given compile options.
 * User is responsible for freeing the string.
 *
 *  @param flags A char * contains all the options provided to compile the
 *  model.
 *  @return string containing the model tag. User is responsible for freeing the
 *  string.
 */
ONNX_MLIR_EXPORT char *omCompileModelTag(const char *flags);

#ifdef __cplusplus
} // namespace onnx_mlir
} // extern C
#endif

#endif
