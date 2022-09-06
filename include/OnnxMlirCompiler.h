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
 *  @return 0 on success or OnnxMlirCompilerErrorCodes on failure.
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
 *  @return 0 on success or OnnxMlirCompilerErrorCodes on failure.
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
 *  @return 0 on success or OnnxMlirCompilerErrorCodes on failure.
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
 *  @param val Value of the option being set.
 *  @return 0 on success or OnnxMlirCompilerErrorCodes error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOption(
    const OptionKind kind, const char *val);

/*!
 *  Clear the compiler option defined by the input parameter.
 */
ONNX_MLIR_EXPORT void omClearCompilerOption(const OptionKind kind);

/*!
 *  Get the compiler options.
 *  @param kind Describe which option kind is being set.
 *  @return A copy of the compiler option string. Caller is responsible for
 *  freeing the returned pointer.
 */
ONNX_MLIR_EXPORT const char *omGetCompilerOption(const OptionKind kind);

/*!
 *  Compile an onnx model from a file containing MLIR or ONNX protobuf. When
 *  generating libraries or jar files, the compiler will link in lightweight
 *  runtimes / jar files. If these libraries / jar files are not in the system
 *  wide directory (typically /usr/local/lib), the user can override the default
 *  location using the ONNX_MLIR_RUNTIME_DIR environment variable.
 *
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  Name may include a path, and must include the file name and its extention.
 *  @param outputBaseName File name without extension to write output.
 *  Name may include a path, must include the file name, and should not include
 * an extention.
 *  @param emissionTarget Target format to compile to.
 *  @param outputFilename Output file name of the compiled output for the given
 * emission target. User is responsible for freeing the string.
 *  @param errorMessage Output error message, if any. User is responsible for
 * freeing the string.
 *  @return 0 on success or OnnxMlirCompilerErrorCodes on failure.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const char **outputFilename, const char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array. When
 *  generating libraries or jar files, the compiler will link in lightweight
 *  runtimes / jar files. If these libraries / jar files are not in the system
 *  wide directory (typically /usr/local/lib), the user can override the default
 *  location using the ONNX_MLIR_RUNTIME_DIR environment variable.
 *
 *  @param inputBuffer ONNX protobuf array.
 *  @param bufferSize Size of ONNX protobuf array.
 *  @param outputBaseName File name without extension to write output.
 *  Name may include a path, must include the file name, and should not include
 * an extention.
 *  @param emissionTarget Target format to compile to.
 *  @param outputFilename Output file name of the compiled output for the given
 * emission target. User is responsible for freeing the string.
 *  @param errorMessage Error message.
 *  @return 0 on success or OnnxMlirCompilerErrorCodes failure. User is
 * responsible for freeing the string.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromArray(const void *inputBuffer,
    int bufferSize, const char *outputBaseName,
    EmissionTargetType emissionTarget, const char **outputFilename,
    const char **errorMessage);

#ifdef __cplusplus
} // namespace onnx_mlir
} // extern C
#endif

#endif
