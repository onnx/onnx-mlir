#!/usr/bin/env python3

##################### common.py ################################################
#
# Copyright 2021-2022 The IBM Research Authors.
#
################################################################################
# Common function `compile_model` called by both
# SignatureExecutionSession and EndiannessAwareExecutionSession
################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import ctypes
import onnx
import subprocess
import variables
from variables import *
import _ctypes


# determine the dynamic input and dim
def determine_dynamic_parameters(test_name):
    if not args.dynamic:
        return None

    # set default value: all inputs, first dimension.
    # Use this script's arguments '--input' and '--dim' or environment variables
    # TEST_INPUT and TEST_DIM to control the values.
    selected_list = {args.input: {args.dim}}
    test_name_cpu = test_name + "_cpu"
    if test_name_cpu in variables.test_for_dynamic:
        if len(variables.test_to_enable_dict[test_name_cpu]) > 1:
            selected_list = variables.test_to_enable_dict[test_name_cpu].get(
                DYNAMIC_SHAPE
            )
    return selected_list


def execute_commands(cmds, dynamic_inputs_dims):
    if args.verbose:
        print(" ".join(cmds), file=sys.stderr)
        print("IMPORTER FORCE DYNAMIC ", dynamic_inputs_dims, file=sys.stderr)

    my_env = os.environ.copy()
    env_string = ""
    if dynamic_inputs_dims is not None:
        first_input = True
        for input_index, dim_indices in dynamic_inputs_dims.items():
            if first_input:
                env_string += str(input_index)
                first_input = False
            else:
                env_string += "|" + str(input_index)
            first_dim = True
            for dim_index in dim_indices:
                if first_dim:
                    env_string += ":" + str(dim_index)
                    first_dim = False
                else:
                    env_string += "," + str(dim_index)
        my_env["IMPORTER_FORCE_DYNAMIC"] = env_string
    subprocess.run(cmds, env=my_env, check=True)


def check_instruction(test_name, exec_name):
    if args.instruction_check and test_name in variables.test_to_enable_symbol_dict:
        symbol_name = variables.test_to_enable_symbol_dict[test_name]
        if symbol_name:
            lib = ctypes.cdll.LoadLibrary(exec_name)
            # Raise AttributeError if symbol undefined
            symbol = getattr(lib, symbol_name)
            _ctypes.dlclose(lib._handle)


def compile_model(model, emit):
    suffix = {"lib": ".so", "obj": ".o", "jni": ".jar"}
    target = {"lib": "--EmitLib", "obj": "--EmitObj", "jni": "--EmitJNI"}
    name = model.graph.name

    # Each model will have its own model_dir. This is necessary for JNI tests
    # since all the models will extract libmodel.so. So if all the models are
    # in the same directory, their libmodel.so will trash each other.
    model_dir = os.path.join(result_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    if args.verbose:
        print("ONNX_HOME=" + os.getenv("ONNX_HOME"), file=sys.stderr)
        print("model_dir=" + model_dir, file=sys.stderr)

    # For node models, write the models in memory out to onnx files.
    # Real (light) models now behave the same (no longer downloaded).
    model_name = os.path.join(model_dir, name + ".onnx")
    # Save model to disk as model_name.onnx.
    onnx.save(model, model_name)

    if args.verbose:
        print(
            ("Success" if os.path.exists(model_name) else "Failure")
            + " saving "
            + model_name,
            file=sys.stderr,
        )

    exec_base = os.path.join(model_dir, name)
    exec_name = exec_base + suffix[emit]
    test_name_cpu = name + "_cpu"

    # Command
    command_list = [TEST_DRIVER]
    if args.Optlevel:
        command_list.append("-O" + args.Optlevel)
    if args.mcpu:  # deprecated
        print("warning, --mcpu option is deprecated, please use --march instead")
        command_list.append("--mcpu=" + args.mcpu)
    if args.march:
        command_list.append("--march=" + args.march)
    if args.mtriple:
        command_list.append("--mtriple=" + args.mtriple)
    if args.maccel:
        command_list.append("--maccel=" + args.maccel)
    if args.input_verification:
        command_list.append("--verifyInputTensors=")
    if args.converter or name in variables.test_need_converter:
        command_list.append("--invokeOnnxVersionConverter=true")
    if args.constants_to_file:
        command_list.append("--store-constants-to-file=true")
    if args.constants_to_file_total_threshold:
        command_list.append(
            "--constants-to-file-total-threshold="
            + str(args.constants_to_file_total_threshold)
        )
    if args.dynamic and test_name_cpu in variables.test_to_enable_dimparams_dict:
        command_list.append(
            "--dimParams=" + variables.test_to_enable_dimparams_dict[test_name_cpu]
        )

    command_list.append(target[emit])
    command_list.append(model_name)
    command_list.append("-o=" + exec_base)

    # Additional args passed in by TEST_COMPILE_ARGS
    # Args are separated by ';'
    additional_args = os.getenv("TEST_COMPILE_ARGS")
    if additional_args is not None:
        compile_args = additional_args.split(";")
        command_list += compile_args

    # Call frontend to process model_name.onnx, bit code will be generated.
    dynamic_inputs_dims = determine_dynamic_parameters(name)
    if args.verbose:
        print("cwd: " + os.getcwd(), file=sys.stderr)
    execute_commands(command_list, dynamic_inputs_dims)

    # Check if compiled model file exists.
    # if not os.path.exists(exec_name):
    #     print("Failed " + TEST_DRIVER + ": " + name, file=sys.stderr)
    #
    # This is not a reliable way of checking if the compilation
    # succeeded or not. Since a compiled model may exist due to
    # a previous successful compilation. The check is now done
    # in execute_commands when calling subprocess.run.

    # Check if specific instruction are included in the compiled model.
    check_instruction(test_name_cpu, exec_name)

    return exec_name
