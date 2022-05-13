# SPDX-License-Identifier: Apache-2.0

##################### CheckONNXModelZoo.py #####################################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################
#
# This script is used to check models in https://github.com/onnx/models.
#
################################################################################

import os
import sys
import argparse
import subprocess
import tempfile
import difflib
from joblib import Parallel, delayed
"""
Note:
    - This script must be invoked from the root folder of https://github.com/onnx/models.
    - This script will call RunONNXModel.py. Make sure to put RunONNXModel.py and this script in the same folder.
    - Environment variable ONNX_MLIR_HOME is needed to find onnx-mlir.
    - By default, the script checks all models in the model zoo.
    - Use `-m model_name` to check one model, or directly edit `models_to_run` to check a list of selected models.

Example:
    $ git clone https://github.com/onnx/models
    $ cd models
    $ ln -s /onnx_mlir/test/onnx-model/test/onnx-model-zoo/CheckONNXModelZoo.py CheckONNXModelZoo.py
    $ ln -s /onnx_mlir/utils/RunONNXModel.py RunONNXModel.py
    $ VERBOSE=1 ONNX_MLIR_HOME=/onnx-mlir/build/Release/ python CheckONNXModelZoo.py -pull-models -m mnist-8 -compile_args="-O3 -mcpu=z14"
"""

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc. sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")
"""
VERBOSE values:
    - 0: turn off
    - 1: user information
    - 2: (user + command) information
"""
VERBOSE = int(os.environ.get('VERBOSE', 0))


def log_l1(*args):
    if (VERBOSE >= 1):
        print(' '.join(args))


def log_l2(*args):
    if (VERBOSE >= 2):
        print(' '.join(args))


"""Commands will be called in this script.
"""
FIND_MODEL_PATHS_CMD = ['find', '.', '-type', 'f', '-name', '*.tar.gz']
# git lfs pull --include="${onnx_model}" --exclude=""
PULL_CMD = ['git', 'lfs', 'pull', '--exclude=\"\"']
# git lfs pointer --file = "${onnx_model}" > ${onnx_model}.pt
CLEAN_CMD = ['git', 'lfs', 'pointer']
# git checkout file_path
CHECKOUT_CMD = ['git', 'checkout']
# tar -xzvf file.tar.gz
UNTAR_CMD = ['tar', '-xzvf']
RM_CMD = ['rm']
MV_CMD = ['mv']
# Compile, run and verify an onnx model.
RUN_ONNX_MODEL = ['python', 'RunONNXModel.py']


def execute_commands(cmds):
    log_l2(' '.join(cmds))
    out = subprocess.Popen(cmds,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    if stderr:
        return (False, stderr.decode("utf-8"))
    else:
        return (True, stdout.decode("utf-8"))


def execute_commands_to_file(cmds, ofile):
    log_l2(' '.join(cmds))
    with open(ofile, 'w') as output:
        server = subprocess.Popen(cmds,
                                  stdout=output,
                                  stderr=subprocess.STDOUT)
        stdout, stderr = server.communicate()


# Deprecated models according to: https://github.com/onnx/models/pull/389
deprecated_models = {
    "mnist-1",
    "bvlcalexnet-3",
    "caffenet-3",
    "densenet-3",
    "inception-v1-3",
    "inception-v2-3",
    "rcnn-ilsvrc13-3",
    "resnet50-caffe2-v1-3",
    "shufflenet-3",
    "zfnet512-3",
    "vgg19-caffe2-3",
    "emotion-ferplus-2",
}

# States
NO_TEST = 0
TEST_FAILED = 1
TEST_PASSED = 2


def obtain_all_model_paths():
    _, model_paths = execute_commands(FIND_MODEL_PATHS_CMD)
    model_paths = model_paths.split('\n')
    # Remove empty paths and prune '._' in a path.
    model_paths = [path[2:] for path in model_paths if path]
    model_names = [
        path.split('/')[-1][:-len(".tag.gz")] for path in model_paths
    ]  # remove .tag.gz
    deprecated_names = set(model_names).intersection(deprecated_models)

    log_l1('\n')
    deprecated_msg = ""
    if (len(deprecated_names) != 0):
        deprecated_msg = "where " + \
            str(len(deprecated_names)) + \
            " models are deprecated (using very old opsets, e.g. <= 3)"
    log_l1("# There are {} models in the ONNX model zoo {}".format(
        len(model_paths), deprecated_msg))
    log_l1("See https://github.com/onnx/models/pull/389",
           "for a list of deprecated models\n")
    return model_names, model_paths


def check_model(model_path, model_name, compile_args):
    passed = NO_TEST
    with tempfile.TemporaryDirectory() as tmpdir:
        # untar
        log_l1('Extracting the .tag.gz to {}'.format(tmpdir))
        execute_commands(UNTAR_CMD + [model_path, '-C', tmpdir])
        _, onnx_files = execute_commands(
            ['find', tmpdir, '-type', 'f', '-name', '*.onnx'])
        # log_l1(onnx_files)
        # temporary folder's structure:
        #   - model.onnx
        #   - test_data_set_0
        #   - test_data_set_1
        #   - test_data_set_2
        #   - ...
        #   - test_data_set_n

        # Check .onnx file.
        if (len(onnx_files) == 0):
            log_l1("There is no .onnx file for this model. Ignored.")
            return NO_TEST
        onnx_file = onnx_files.split('\n')[0]

        # Check data sets.
        has_data_sets = False
        _, data_sets = execute_commands(
            ['find', tmpdir, '-type', 'd', '-name', 'test_data_set*'])
        if (len(data_sets) > 0):
            has_data_sets = True
            data_set = data_sets.split('\n')[0]
        else:
            # if there is no `test_data_set` subfolder, find a folder containing .pb files.
            _, pb_files = execute_commands(
                ['find', tmpdir, '-name', '*.pb', '-printf', '%h\n'])
            if (len(pb_files) > 0):
                has_data_sets = True
                data_set = pb_files.split('\n')[0]
        if (not has_data_sets):
            log_l1("Warning: This model does not have test data sets.")

        # compile, run and verify.
        log_l1("Checking the model {} ...".format(model_name))
        compile_options = "--compile_args=" + (compile_args
                                               if compile_args else "-O3")
        options = [compile_options]
        if has_data_sets:
            options += ['--verify=ref']
            options += ['--data_folder={}'.format(data_set)]
        ok, msg = execute_commands(RUN_ONNX_MODEL + [onnx_file] + options)
        state = TEST_PASSED if ok else TEST_FAILED
        log_l1(msg)
    return state


def pull_and_check_model(model_path, compile_args, pull_models, keep_model):
    state = NO_TEST

    # Ignore deprecated models.
    model_name = model_path.split('/')[-1][:-len(".tag.gz")]  # remove .tag.gz
    if model_name in deprecated_models:
        log_l1("The model {} is deprecated. Ignored.".format(model_name))
        return state, model_name

    # pull the model.
    if pull_models:
        log_l1('Downloading {}'.format(model_path))
        pull_cmd = PULL_CMD + ['--include={}'.format(model_path)]
        ok, _ = execute_commands(pull_cmd)
        if not ok:
            log_l1("Failed to pull the model {}. Ignored.".format(model_name))

    # check the model.
    state = check_model(model_path, model_name, compile_args)

    if pull_models and (not keep_model):
        # remove the model to save the storage space.
        clean_cmd = CLEAN_CMD + ['--file={}'.format(model_path)]
        execute_commands_to_file(clean_cmd, '{}.pt'.format(model_path))
        execute_commands(RM_CMD + [model_path])
        execute_commands(MV_CMD + ['{}.pt'.format(model_path), model_path])
        execute_commands(CHECKOUT_CMD + [model_path])

    return state, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '-model',
        metavar='model_name',
        help="Only process a single model in the ONNX model zoo."
        " Passing the name of the model, e.g. mnist-8."
        " Use -p to know model names.")
    parser.add_argument('-p',
                        '-print_model_paths',
                        action='store_true',
                        help="Only print model paths in the model zoo.")
    parser.add_argument('-k',
                        '-keep_pulled_models',
                        action='store_true',
                        help="Keep the pulled models")
    parser.add_argument('-a',
                        '-assertion',
                        action='store_true',
                        help="Raise assertion if there are failed models")
    parser.add_argument(
        '-compile_args',
        help="Options passing to onnx-mlir to compile a model.")
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        '-njobs',
        type=int,
        default=1,
        help="The number of processes in parallel."
        " The large -njobs is, the more disk space is needed"
        " for downloaded onnx models. Default 1.")
    parallel_group.add_argument(
        '-pull_models',
        action='store_true',
        help="Pull models from the remote git repository."
        " This requires git-lfs. Please follow the instruction here to install"
        " git-lfs: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage."
    )
    args = parser.parse_args()

    # Collect all model paths in the model zoo
    all_model_names, all_model_paths = obtain_all_model_paths()
    if (args.p):
        for path in all_model_paths:
            print(path)
        return

    # By default, run all models in the model zoo.
    # But, if `-s` is specified, only run a model if given.
    models_to_run = all_model_names

    # If we would like to run with models of interest only, set models_to_run.
    # models_to_run = ['mnist-8', 'yolov4', 'resnet50-v2-7']

    if (args.m):
        models_to_run = [args.m]

    target_model_paths = []
    for name in models_to_run:
        if name not in all_model_names:
            print(
                "Model", args.m,
                "not found. Do you mean one of the following? ",
                difflib.get_close_matches(name, all_model_names,
                                          len(all_model_names)))
            return
        target_model_paths += [m for m in all_model_paths if name in m]

    # Start processing the models.
    results = Parallel(n_jobs=args.njobs,
                       verbose=1)(delayed(pull_and_check_model)(
                           path, args.compile_args, args.pull_models, args.k)
                                  for path in target_model_paths)

    # Report the results.
    tested_models = [r[1] for r in results if r[0] != NO_TEST]
    print("{} models tested: {}\n".format(len(tested_models),
                                          ', '.join(tested_models)))
    passed_models = [r[1] for r in results if r[0] == TEST_PASSED]
    print("{} models passed: {}\n".format(len(passed_models),
                                          ', '.join(passed_models)))
    if len(passed_models) != len(tested_models):
        failed_models = [r[1] for r in results if r[0] == TEST_FAILED]
        msg = "{} model failed: {}\n".format(len(failed_models),
                                             ', '.join(failed_models))
        if args.assertion:
            raise AssertionError(msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()
