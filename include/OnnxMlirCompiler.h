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
#define ONNX_MLIR_NO_EXPORT
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

struct OMCompilerOptions;
#ifndef _cplusplus
typedef struct OMCompilerOptions OMCompilerOptions;
#endif

/*!
 * Type that contains the defined compiler options and their values.
 */

/*!
 * Create a list of compiler options. All options are set to undefined.
 * @return a pointer to the compiler option datastructure, null in clase
 * of error.
 */

ONNX_MLIR_EXPORT OMCompilerOptions *omCreateCompilerOptions();

/*!
 *  Destoy a list of compiler options.
 *  @param options List to be destroyed.
 */
ONNX_MLIR_EXPORT void omDestroyCompilerOptions(OMCompilerOptions *options);

/*!
 *  Overwrite the given set of compiler options with options defined
 *  by environment variables.
 *  @param options List to be overwridden by environment variable values.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromEnv(
    OMCompilerOptions *options);

/*! 
 *  Overwrite the given set of compiler options with the options defined
 *  by the argc/argv parameters. Values not recoginzed as compiler options
 *  are simply ignored.
 *  @param options List to be overwridden by environment variable values.
 *  @param argc Number of input parameters in argv.
 *  @param argv Array of strings, some of which may be compiler options.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptionsFromArgs(
    OMCompilerOptions *options, const int64_t argc, const char *argv[]);

/*! 
 *  Overwrite the given set of compiler options with the options defined
 *  by input parameters. Values not recoginzed as compiler options
 *  are simply ignored.
 *  @param options List to be overwridden by environment variable values.
 *  @param kind Describe which option kind is beign set.
 *  @param val Value of the option being set. Null pointer undefines the
 *  option.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omSetCompilerOptions(OMCompilerOptions *options, 
    const OptionKind kind, const char *val);

/*!
 *  Compile an onnx model from a file containing MLIR or ONNX protobuf.
 *  @param inputFilename File name pointing onnx model protobuf or MLIR.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param options List of compiler options.
 *  @param errorMessage Error message.
 *  @return 0 on success or non-zero error code on failure.
 */
ONNX_MLIR_EXPORT int64_t omCompileFromFile(const char *inputFilename,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const OMCompilerOptions *options, const char **errorMessage);

/*!
 *  Compile an onnx model from an ONNX protobuf array.
 *  @param inputBuffer ONNX protobuf array.
 *  @param bufferSize Size of ONNX protobuf array.
 *  @param outputBaseName File name without extension to write output.
 *  @param emissionTarget Target format to compile to.
 *  @param options List of compiler options.
 *  @return 0 on success or non-zero error code on failure
 */
ONNX_MLIR_EXPORT int omCompileFromArray(const void *inputBuffer, int bufferSize,
    const char *outputBaseName, EmissionTargetType emissionTarget,
    const OMCompilerOptions *options);

#ifdef __cplusplus
} // namespace onnx_mlir
} // extern C
#endif

#endif
