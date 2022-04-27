import os
import sys
import onnx
import argparse
import subprocess
import tempfile
import difflib
"""
This script is used to check models in https://github.com/onnx/models.
It automatically downloads all models from onnx/models, compiles, runs, and
verifies the models. 

Note:
    - This script must be invoked from the root folder of https://github.com/onnx/models.
    - This script requires git-lfs to download models. Please follow the instruction here to install git-lfs: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage 
    - Environment variable ONNX_MLIR_HOME is needed to find onnx-mlir.
"""

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")

VERBOSE = os.environ.get('VERBOSE', False)

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


def execute_commands(cmds):
    if (VERBOSE):
        print(cmds)
    out = subprocess.Popen(cmds,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    if stderr:
        return (False, stderr.decode("utf-8"))
    else:
        return (True, stdout.decode("utf-8"))


def execute_commands_to_file(cmds, ofile):
    if (VERBOSE):
        print(cmds)
    with open(ofile, 'w') as output:
        server = subprocess.Popen(cmds,
                                  stdout=output,
                                  stderr=subprocess.STDOUT)
        stdout, stderr = server.communicate()


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

RUN_ONNX_MODEL = ['python', '/home/tungld/dl/onnx-mlir/utils/RunONNXModel.py']


def obtain_all_model_paths():
    _, model_paths = execute_commands(FIND_MODEL_PATHS_CMD)
    model_paths = model_paths.split('\n')
    # Remove empty paths and prune '._' in a path.
    model_paths = [path[2:] for path in model_paths if path]
    model_names = [path.split('/')[-1].split('.')[-3] for path in model_paths]
    deprecated_names = set(model_names).intersection(deprecated_models)

    print('\n')
    deprecated_msg = ""
    if (len(deprecated_names) != 0):
        deprecated_msg = "where " + \
            str(len(deprecated_names)) + \
            " models are deprecated (using very old opsets, e.g. <= 3)"
    print("# There are {} models in the ONNX model zoo {}".format(
        len(model_paths), deprecated_msg))
    print("See https://github.com/onnx/models/pull/389",
          "for a list of deprecated models\n")
    return model_names, model_paths


def pull_and_get_ops_from_model_zoo(model_paths, keep_model=False):
    # Read each model in the zoo.
    passed_models = 0
    total_models = 0
    for path in model_paths:
        model_name = path.split('/')[-1].split('.')[-3]
        # Ignore deprecated models.
        if model_name in deprecated_models:
            print("This model is deprecated")
            continue
        print('Downloading {}'.format(path))
        # pull the model.
        pull_cmd = PULL_CMD + ['--include={}'.format(path)]
        execute_commands(pull_cmd)
        with tempfile.TemporaryDirectory() as tmpdir:
            # untar
            print('Extracting the .tag.gz to {}'.format(tmpdir))
            execute_commands(UNTAR_CMD +
                             [path, '-C', tmpdir, '--strip-components=1'])
            _, onnx_models = execute_commands(
                ['find', tmpdir, '-type', 'f', '-name', '*.onnx'])
            # print(onnx_models)
            # temporary folder's structure:
            #   - model.onnx
            #   - test_data_set_0
            #   - test_data_set_1
            #   - test_data_set_2
            #   - ...
            #   - test_data_set_n
            onnx_model = onnx_models.split('\n')[0]
            data_set = tmpdir + '/test_data_set_0'

            # compile, run and verify.
            total_models += 1
            options = ['--compile_args=-O3 --mcpu=z14']
            options += ['--verify=ref']
            options += ['--data_folder={}'.format(data_set)]
            passed, msg = execute_commands(RUN_ONNX_MODEL + [onnx_model] + options)
            print(msg)
            if passed:
                passed_models += 1

        if not keep_model:
            # remove the model to save the storage space.
            clean_cmd = CLEAN_CMD + ['--file={}'.format(path)]
            execute_commands_to_file(clean_cmd, '{}.pt'.format(path))
            execute_commands(RM_CMD + [path])
            execute_commands(MV_CMD + ['{}.pt'.format(path), path])
            execute_commands(CHECKOUT_CMD + [path])

    print("Tested {} models.".format(total_models))
    print("{} models passed.".format(passed_models))

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-s',
        metavar='model_name',
        help="Only process a single model in the ONNX model zoo."
        "Passing the name of the model, e.g. mnist-8."
        "Use -p to know model names.")
    group.add_argument('-p',
                       action='store_true',
                       help="only print model paths in the model zoo.")
    parser.add_argument('-k',
                        action='store_true',
                        help="keep the downloaded model")
    args = parser.parse_args()

    # Collect all model paths in the model zoo
    all_model_names, all_model_paths = obtain_all_model_paths()
    if (args.p):
        for path in all_model_paths:
            print(path)
        return

    # By default, run all models in the model zoo.
    # But, if `-p` is specified, only run a model if given.
    models_to_run = all_model_names

    # If we would like to run with models of interest only, set models_to_run.
    # models_to_run = ['mnist-8', 'yolov4', 'resnet50-v2-7']

    if (args.s):
        models_to_run = [args.s]

    target_model_paths = []
    for name in models_to_run:
        if name not in all_model_names:
            print(
                "Model", args.s,
                "not found. Do you mean one of the following? ",
                difflib.get_close_matches(name, all_model_names,
                                          len(all_model_names)))
            return
        target_model_paths += [m for m in all_model_paths if name in m]

    # Start processing the models.
    pull_and_get_ops_from_model_zoo(target_model_paths, args.k)


if __name__ == "__main__":
    main()
