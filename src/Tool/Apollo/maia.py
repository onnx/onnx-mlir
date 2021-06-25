#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import platform
import re

#   ["command",          "output postfix",            "arg1", "arg2", ...                                                                                                             ]
passes = [
    ["onnx-mlir",        ".mlir",                     "--EmitMLIR", "--npu", "--preserveLocations"                                                                                    ],

  # TVP kernel outlining.
  # canonicalize is called to remove the empty affine loops we may have picked up
    ["mlir-opt",         ".normalize.mlir",           "--affine-loop-normalize", "--symbol-dce", "--canonicalize", "--allow-unregistered-dialect", "-mlir-print-debuginfo"            ],
    ["mlir-opt",         ".affine.for.2.tvp.mlir",    "--convert-affine-for-to-tvp", "--allow-unregistered-dialect", "-mlir-print-debuginfo"                                          ],
    ["mlir-opt",         ".krnl.outlining.mlir",      "--tvp-kernel-outlining", "--allow-unregistered-dialect"                                                                        ],
    ["mlir-opt",         ".krnl.generation.mlir",      "--generate-tvp-kernels", "--assign-kernel-ids", "--allow-unregistered-dialect"                                                ],

  # TCP passes with access to the full graph. Has dependency on TVP kernel body.
    ["mlir-opt",         ".nepal.dma.insertion.mlir", "--nepal-dma-insertion", "--allow-unregistered-dialect"                                                                         ],
    ["mlir-opt",         ".nepal.dma.optimize.mlir",  "--nepal-dma-optimize", "--allow-unregistered-dialect"                                                                          ],
    ["mlir-opt",         ".affine.dma.to.nepal.mlir", "--affine-dma-to-nepal", "--allow-unregistered-dialect"                                                                         ],
    ["mlir-opt",         ".tcp.canon.args.mlir",      "--create-nepal-arguments", "--allow-unregistered-dialect", "--apply-artemis-calling-convention"                                ],
    # cse is run because some unused floats stick around that the NIOS processor isn't equipped to handle
    ["mlir-opt",         ".tcp.canon.full.mlir",      "--pass-pipeline=func(lower-affine,cse)", "--allow-unregistered-dialect"                                                        ],
    ["mlir-opt",         ".post.cpp.mlir",            "--nepal-generation=output-file-name=tcp_driver", "--allow-unregistered-dialect"                                                ],

  # Remove TCP IR and only keep TVP kernels.
    ["mlir-opt",         ".tvp.only.mlir",            "--tvp-kernel-filter", "--allow-unregistered-dialect"                                                                           ],

  # TVP passes with access only to TVP kernels.
    ["mlir-opt",         ".memspace.removal.mlir",    "--tvp-kernel-memspace-removal"                                                                                                 ],
    ["mlir-opt",         ".sv.mlir",                  "--pass-pipeline=func(affine-super-vectorize{virtual-vector-size=%SIZE%})"                                                      ],
    ["mlir-opt",         ".lower.affine.mlir",        "--lower-affine"                                                                                                                ],
    ["mlir-opt",         ".vector.2.tvp.mlir",        "--convert-vector-to-tvp"                                                                                                       ],
    ["mlir-opt",         ".scf.2.std.mlir",           "--convert-scf-to-std"                                                                                                          ],
    ["mlir-opt",         ".std.2.tvp.mlir",           "--convert-std-to-tvp"                                                                                                          ],
    ["mlir-opt",         ".vector.2.scf.mlir",        "--convert-vector-to-scf"                                                                                                       ],
    ["mlir-opt",         ".tvp.2.llvm.mlir",          "--convert-tvp-to-llvm"                                                                                                         ],
    ["mlir-opt",         ".dispatcher.llvm.mlir",     "--llvm-generate-dispatcher"                                                                                                    ],
    ["mlir-translate",   ".ll",                       "--mlir-to-llvmir"                                                                                                              ],
    ["llc",              ".s",                        "-mtriple=apollo-none-none -max-jump-table-size=0 -filetype=asm -O2"                                                            ],
]

if platform.system() == "Windows":
    onnx_mlir_cmd = "onnx-mlir.exe"
    mlir_opt_cmd = "mlir-opt.exe"
    mlir_translate_cmd = "mlir-translate.exe"
    llc_cmd = "llc.exe"
else:
    onnx_mlir_cmd = "onnx-mlir"
    mlir_opt_cmd = "mlir-opt"
    mlir_translate_cmd = "mlir-translate"
    llc_cmd = "llc"

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    exe_file = os.path.join(os.getcwd(), program)
    if is_exe(exe_file):
       return exe_file

    return None

def print_pass(pass_to_print, pass_index, input, output):
    print("Pass " + str(pass_index) + ": ", end='')
    for j, field in enumerate(pass_to_print):
        if j != 1:
            print(field + ' ', end='')
    print(input, '>', output)

def print_all_passes():
    for i, curr_pass in enumerate(passes):
        print_pass(curr_pass, i, "input_file", "output_file")

def run_passes(passes, args, first_input, vector_length, from_pass, to_pass, subtarget):
    outputFileBase = os.path.join(os.path.curdir, os.path.basename(args.f))
    for i, curr_pass in enumerate(passes):
        command = curr_pass[0]

        if i == from_pass:
            input = first_input
        else:
            input = outputFileBase + passes[i-1][1]

        if i < from_pass:
            continue

        if i > to_pass:
            break

        command_list = curr_pass.copy()
        if command_list[0] == "onnx-mlir":
            command_list[0] = onnx_mlir_cmd
        elif command_list[0] == "mlir-opt":
            command_list[0] = mlir_opt_cmd
        elif command_list[0] == "mlir-translate":
            command_list[0] = mlir_translate_cmd
        elif command_list[0] == "llc":
            command_list[0] = llc_cmd

        command_list[1] = input

        command_list[2] = command_list[2].replace("%SIZE%", str(vector_length))

        convert_std_to_tvp = "convert-std-to-tvp" if subtarget == 'apollo' else 'convert-std-to-tvp=' + subtarget
        command_list[2] = command_list[2].replace("convert-std-to-tvp", convert_std_to_tvp)
        if subtarget == 'athena':
            # set target triple for llc to 'athena-none-none instead of default 'apollo-none-none'
            command_list[2] = command_list[2].replace('-mtriple=apollo', '-mtriple=' + subtarget)

        # Set the correct TTU command type
        if subtarget == 'apollo':
            if args.ttu_type2:
                command_list[2] = command_list[2].replace('nepal-generation=', 'nepal-generation=' + 'target=Apollo ' + 'TTU-command-type=2 ')
            else:
                command_list[2] = command_list[2].replace('nepal-generation=', 'nepal-generation=' + 'target=Apollo ' + 'TTU-command-type=1 ')

        output = os.devnull
        if command == "onnx-mlir":
            outputFileNoExt, _ = os.path.splitext(outputFileBase)
            command_list.append(str("-o=" + outputFileNoExt))
        elif ( command == "mlir-opt" or command == "mlir-translate" or command == "llc" ) :
            command_list.append(str("-o=" + outputFileBase + curr_pass[1]))
        else:         
            output = outputFileBase + curr_pass[1]

        print_pass(command_list, i, input, output)
        outputF = open(output, "w+")
        retCode = subprocess.call(command_list, stdout=outputF)

        if retCode != 0:
            exit(retCode)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--f', '--file', type=str, required=False, help='input file')
    group.add_argument('--p', '--print_passes', action='store_true', help='print pass pipeline')
    parser.add_argument('--from_pass', type=int, required=False, help='set the beginning pass of the pipeline')
    parser.add_argument('--to_pass', type=int, required=False, help='set the ending pass of the pipeline')
    parser.add_argument('--om', '--onnx-mlir', type=str, required=False, help='set the path to onnx-mlir')
    parser.add_argument('--mo', '--mlir-opt', type=str, required=False, help='set the path to mlir-opt')
    parser.add_argument('--mt', '--mlir-translate', type=str, required=False, help='set the path to mlir-translate')
    parser.add_argument('--llc', type=str, required=False, help='set the path to llc')
    parser.add_argument('--vector_length', type=int, required=False, help='vector length override for supervectorizer pass')
    parser.add_argument('--omb', '--onnx-mlir-bin', type=str, required=False, help='set the path to onnx-mlir binaries')
    parser.add_argument('--mb', '--mlir-bin', type=str, required=False, help='set the path to mlir binaries')
    parser.add_argument('--athena', type=str2bool, nargs='?', const=True, default=False, required=False, help='Lower to LLVM for Athena')
    parser.add_argument('--ttu_type2', action='store_true', help='Use Type 2 TTU commands')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    global onnx_mlir_cmd
    global mlir_translate_cmd
    global mlir_opt_cmd
    global llc_cmd
    
    if args.p == True:
        print_all_passes()
        sys.exit(0)

    if args.om != None:
        onnx_mlir_cmd = args.om

    if args.mo != None:
        mlir_opt_cmd = args.mo

    if args.mt != None:
        mlir_translate_cmd = args.mt

    if args.llc != None:
        llc_cmd = args.llc

    if args.vector_length != None:
        vector_length = args.vector_length
    else:
        vector_length = 256

    if (args.athena == True):
        subtarget = 'athena'
    else:
        subtarget = 'apollo'

    if (args.omb != None):
        onnx_mlir_cmd = os.path.join(args.omb, onnx_mlir_cmd)

    if (args.mb != None):
        mlir_opt_cmd = os.path.join(args.mb, mlir_opt_cmd)
        mlir_translate_cmd = os.path.join(args.mb, mlir_translate_cmd)
        llc_cmd = os.path.join(args.mb, llc_cmd)

    if which(onnx_mlir_cmd) == None:
        sys.exit("onnx-mlir is not available")
    if which(mlir_opt_cmd) == None:
        sys.exit("mlir-opt is not available")
    if which(mlir_translate_cmd) == None:
        sys.exit("mlir-translate is not available")
    if which(llc_cmd) == None:
        sys.exit("llc is not available")

    from_pass = 0 if args.from_pass == None else args.from_pass
    to_pass = len(passes)-1 if args.to_pass == None else args.to_pass

    run_passes(passes, args, args.f, vector_length, from_pass, to_pass, subtarget)

if __name__ == "__main__":
    main()
