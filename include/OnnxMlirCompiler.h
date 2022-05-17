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
 *  Define ONNX-MLIR compiler options with options defined by
 *  the envVarName (default ONNX_MLIR_FLAGS) environment variable.
 *  Values not recognized as compiler options result in an error.
 *  Only a single call to omSetCompilerOptionsFromEnv,
 *  omSetCompilerOptionsFromArgs, or omSetCompilerOptionsFromEnvAndArgs
 *  is allowed.
 *  @param envVarName Environment variable name, use default when null.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnv(const char *envVarName);

/*!
 *  Define ONNX-MLIR compiler options with options defined by
 *  the argc/argv parameters. Call expects argv[0] to contain the program
 *  name. Values not recognized as compiler options result in an error.
 *  Only a single call to omSetCompilerOptionsFromEnv,
 *  omSetCompilerOptionsFromArgs, or omSetCompilerOptionsFromEnvAndArgs
 *  is allowed.
 *  @param argc Number of input parameters in argv.
 *  @param argv Array of strings, some of which may be compiler options.
 *  First argv is ignored as it contains the name of the program.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgs(
    int64_t argc, char *argv[]);

/*!
 *  Define ONNX-MLIR compiler options with options defined by
 *  the envVarName (default ONNX_MLIR_FLAGS) environment variable
 *  and the argc/argv parameters. Call expects argv[0] to contain the program
 *  name. Values not recognized as compiler options result in an error.
 *  Only a single call to omSetCompilerOptionsFromEnv,
 *  omSetCompilerOptionsFromArgs, or omSetCompilerOptionsFromEnvAndArgs
 *  is allowed.
 *  @param argc Number of input parameters in argv.
 *  @param argv Array of strings, some of which may be compiler options.
 *  First argv is ignored as it contains the name of the program.
 *  @param envVarName Environment variable name, use default when null.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgsAndEnv(
    int64_t argc, char *argv[], const char *envVarName);

/*!
 *  Overwrite the compiler option defined by input parameters.
 *  Set default value by calling this function before a call to
 *  omSetCompilerOptionsFromEnv, omSetCompilerOptionsFromArgs, or
 *  omSetCompilerOptionsFromEnvAndArgs. Or overwrite the current value
 *  by calling this function after one of the above 3 setter functions.
 *  @param kind Describe which option kind is being set.
 *  @param val Value of the option being set. Empty string reset the
 *  option. Setting TargetAccel is different. When passing a valid
 *  accelerator, it is added to the list of target accelerators unless
 *  val="RESET", in which case the list is cleared.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOption(
    const OptionKind kind, const char *val);

/*!
 *  Get the compiler options.
 *  @param kind Describe which option kind is being set.
 *  @return A copy of the compiler option string. Caller is responsible for
 *  freeing the returned pointer.
 */
ONNX_MLIR_EXPORT const char *omGetCompilerOption(const OptionKind kind);

/*!
 *  Compile an onnx model from a file containing MLIR or ONNX protobuf.
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param errorMessage Error message.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array.
 *  @param inputBuffer ONNX protobuf array.
 *  @param bufferSize Size of ONNX protobuf array.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param errorMessage Error message.
 *  @return 0 on success or non-zero error code on failure
 */
ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget, const char **errorMessage);

#ifdef __cplusplus
} // namespace onnx_mlir
} // extern C
#endif

#endif
