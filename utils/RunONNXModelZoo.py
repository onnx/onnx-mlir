#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

####################### RunONNXModelZoo.py #####################################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################
#
# This script is used to check models in https://github.com/onnx/models.
#
################################################################################

import argparse
import difflib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

from datetime import datetime
from joblib import Parallel, delayed
from pathlib import Path
from urllib.parse import urlsplit

"""
Note:
    - This script will clone https://github.com/onnx/models or reset the local repo.
    - This script will call RunONNXModel.py. Make sure to put RunONNXModel.py and this script in the same folder.
    - Environment variable ONNX_MLIR_HOME is needed to find onnx-mlir.
    - By default, the script checks all models in the model zoo.
    - Use `-m model_name` to check a list of selected models.

Example:
    $ ONNX_MLIR_HOME=/onnx-mlir/build/Release/ /onnx-mlir/utils/RunONNXModelZoo.py -m mnist-8 -c "-O3 --march=z16"
"""

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc. sub-folders in which ONNX-MLIR "
        "executables and libraries can be found."
    )

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
# For Parallel verbose
VERBOSITY_LEVEL = {"debug": 10, "info": 5, "warning": 1, "error": 0, "critical": 0}

ONNX_MODEL_ZOO_URL = "https://github.com/onnx/models"
ONNX_MODEL_ZOO_DOWNLOAD = ONNX_MODEL_ZOO_URL + "/raw/main"

"""Commands will be called in this script.
"""

# modelzoo has been completely restructured and the original models are now under
# the "validated" directory. We could check all the new models as well but that
# would take very long (about 6 hours on the Jenkins CI) so we still only check
# the original models under "validated".
FIND_MODEL_PATHS_CMD = ["find", "validated", "-type", "f", "-name", "*.tar.gz"]
GIT_CMD = ["git"]
# Use curl instead of wget since most systems have curl preinstalled
# and curl is more flexible than wget
CURL_CMD = ["curl", "--insecure", "--retry", "50", "--location", "--silent"]

# RunONNXModel.py is assumed to be in the same directory where
# RunONNXModelZoo.py is (sys.path[0])
RUN_ONNX_MODEL_CMD = [os.path.join(sys.path[0], "RunONNXModel.py")]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--compile-args", help="Options passing to onnx-mlir to compile a model."
    )
    parser.add_argument(
        "-C", "--compile-only", action="store_true", help="Only compile models."
    )
    parser.add_argument(
        "-f",
        "--force-clean",
        action="store_true",
        default=False,
        help="Force clean existing model zoo repo.",
    )
    parser.add_argument(
        "-H",
        "--Html",
        default=None,
        const="modelzoo.html",
        action="store",
        nargs="?",
        help="Generate model zoo test report in html.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="The number of processes in parallel."
        " The larger JOBS is, the more disk space is needed"
        " for downloaded onnx models. Default 1.",
    )
    parser.add_argument(
        "-k", "--keep-models", action="store_true", help="Keep the pulled models"
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="log level, default info",
    )
    parser.add_argument(
        "--log-to-file",
        action="store",
        nargs="?",
        const="compilation.log",
        default=None,
        help="Output compilation messages to file, default compilation.log",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="model_name",
        help="Only process a list of models in the ONNX model zoo."
        " Passing the name of the models, e.g. 'mnist-8 yolov4'."
        " Use -p to know model names. Without -m, the script "
        " checks all models in the model zoo.",
    )
    parser.add_argument(
        "-p",
        "--print-paths",
        action="store_true",
        help="Only print model paths in the model zoo.",
    )
    parser.add_argument(
        "-q",
        "--historydir",
        default="",
        help="History dir for previously published results, no default.",
    )
    parser.add_argument(
        "-r",
        "--reportdir",
        default=os.getcwd(),
        help="Report dir for generating tests results, default cwd.",
    )
    parser.add_argument(
        "-w",
        "--workdir",
        default=os.getcwd(),
        help="Work dir for cloning and downloading, default cwd.",
    )
    return parser.parse_args()


# log to stderr so that stdout can be used for check results
def get_logger():
    logging.basicConfig(
        stream=sys.stderr,
        level=LOG_LEVEL[args.log_level],
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger("RunONNXModelZoo.py")


args = get_args()
logger = get_logger()


def execute_commands(cmds, cwd=None, tmout=None):
    logger.debug("cmd={} cwd={}".format(" ".join(cmds), cwd))
    out = subprocess.Popen(
        cmds, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        stdout, stderr = out.communicate(timeout=tmout)
    except subprocess.TimeoutExpired:
        # Kill the child process and finish communication
        out.kill()
        stdout, stderr = out.communicate()
        return (
            False,
            (
                stderr.decode("utf-8")
                + stdout.decode("utf-8")
                + "Timeout after {} seconds".format(tmout)
            ),
        )
    msg = stderr.decode("utf-8") + stdout.decode("utf-8")
    if out.returncode == -signal.SIGSEGV:
        return (False, msg + "Segfault")
    if out.returncode != 0:
        return (False, msg + "Return code {}".format(out.returncode))
    return (True, stdout.decode("utf-8"))


def execute_commands_to_file(cmds, ofile, cwd=None):
    logger.debug(" ".join(cmds))
    with open(ofile, "w") as output:
        server = subprocess.Popen(
            cmds, cwd=cwd, stdout=output, stderr=subprocess.STDOUT
        )
        stdout, stderr = server.communicate()


# Deprecated models according to: https://github.com/onnx/models/pull/389
deprecated_models = {
    "bvlcalexnet-3",
    "caffenet-3",
    "densenet-3",
    "emotion-ferplus-2",
    "inception-v1-3",
    "inception-v2-3",
    "mnist-1",
    "rcnn-ilsvrc13-3",
    "resnet50-caffe2-v1-3",
    "shufflenet-3",
    "vgg19-caffe2-3",
    "zfnet512-3",
}

int8_models = {
    "bvlcalexnet-12-int8",
    "caffenet-12-int8",
    "densenet-12-int8",
    "efficientnet-lite4-11-int8",
    "FasterRCNN-12-int8",
    "fcn-resnet50-12-int8",
    "googlenet-12-int8",
    "inception-v1-12-int8",
    "MaskRCNN-12-int8",
    "mnist-12-int8",
    "mobilenetv2-12-int8",
    "resnet50-v1-12-int8",
    "ResNet101-DUC-12-int8",
    "shufflenet-v2-12-int8",
    "ssd-12-int8",
    "ssd_mobilenet_v1_12-int8",
    "squeezenet1.0-12-int8",
    "vgg16-12-int8",
    "yolov3-12-int8",
    "zfnet512-12-int8",
}

excluded_models = deprecated_models.union(int8_models)

# Additional information passed to RunONNXModel.py.
# For example: "t5-encoder-12": ['--shape-info=0:1x2,1:1x2x768']
RunONNXModel_additional_options = {}

# States
TEST_SKIPPED = 0
TEST_FAILED = 1
TEST_PASSED = 2


# Clone hypershift and related source repos
def clone_modelzoo_source(repo_url, work_dir):
    repo_dir = os.path.join(work_dir, Path(urlsplit(repo_url).path).stem)

    # Remove repo directory if -f|--force-clean specified
    if args.force_clean and Path(repo_dir).exists():
        logger.debug("repo {} force cleaned".format(repo_dir))
        shutil.rmtree(repo_dir)

    # If .git exists, assume repo already cloned
    if (Path(repo_dir) / ".git").is_dir():
        logger.debug("repo {} reset".format(repo_dir))
        execute_commands(GIT_CMD + ["reset", "--hard"], cwd=repo_dir)
        execute_commands(GIT_CMD + ["clean", "-xdf"], cwd=repo_dir)
    # Either repo doesn't exist, or it's invalid
    else:
        if Path(repo_dir).exists():
            logger.error("repo {} not a git repo, no overwrite".format(repo_dir))
            return None

        logger.debug("clone into {}".format(repo_dir))
        execute_commands(GIT_CMD + ["clone", repo_url, repo_dir])

    return repo_dir


# It would have been much simpler if the ONNX_HUB_MANIFEST.json
# has been kept up to date.
def obtain_all_model_paths(repo_dir):
    _, model_paths = execute_commands(FIND_MODEL_PATHS_CMD, cwd=repo_dir)
    model_paths = model_paths.split("\n")
    # Remove empty paths and prune './' in a path.
    model_paths = [
        (path[2:] if path.startswith("./") else path) for path in model_paths if path
    ]
    model_names = [
        path.split("/")[-1][: -len(".tar.gz")] for path in model_paths
    ]  # remove .tar.gz
    excluded_names = set(model_names).intersection(excluded_models)

    excluded_msg = ""
    if len(excluded_names) != 0:
        excluded_msg = (
            " where "
            + str(len(excluded_names))
            + " models are not checked because of old opsets or quantization"
        )
    logger.debug(
        "There are {} models in the ONNX model zoo{}.".format(
            len(model_paths), excluded_msg
        )
    )
    return model_names, model_paths


def check_model(model_path, model_name, compile_args, report_dir):
    passed = TEST_SKIPPED
    with tempfile.TemporaryDirectory() as tmpdir:
        # untar
        logger.debug("Extracting the .tar.gz to {}".format(tmpdir))
        with tarfile.open(model_path, "r:gz") as tgz:
            tgz.extractall(tmpdir)
        # ignore files starting with "." created by Mac OSX!
        _, onnx_files = execute_commands(
            ["find", tmpdir, "-type", "f", "-name", "[^.]*.onnx"]
        )
        # logger.debug(onnx_files)
        # temporary folder's structure:
        #   - model.onnx
        #   - test_data_set_0
        #   - test_data_set_1
        #   - test_data_set_2
        #   - ...
        #   - test_data_set_n

        # Check .onnx file.
        if len(onnx_files) == 0:
            logger.warning("There is no .onnx file for this model. Ignored.")
            return TEST_SKIPPED
        onnx_file = onnx_files.split("\n")[0]

        # Check data sets.
        has_data_sets = False
        _, data_sets = execute_commands(
            ["find", tmpdir, "-type", "d", "-name", "test_data_set*"]
        )
        data_sets_list = [s for s in data_sets.split("\n") if s]
        if len(data_sets_list) > 0:
            has_data_sets = True
            # Sort the list to get test_data_set_0 by default since other data
            # sets are sometimes ill-formed.
            data_sets_list.sort()
            data_set = data_sets_list[0]
        else:
            # if there is no `test_data_set` subfolder, find a folder containing .pb files.
            _, pb_files = execute_commands(
                ["find", tmpdir, "-name", "*.pb", "-printf", "%h\n"]
            )
            if len(pb_files) > 0:
                has_data_sets = True
                data_set = pb_files.split("\n")[0]
        if not has_data_sets:
            logger.warning(
                "The model {} does not have test data sets. Will check the model with random data.".format(
                    model_name
                )
            )

        # compile, run and verify.
        logger.debug("Checking the model {} ...".format(model_name))
        compile_options = "--compile-args=" + (compile_args if compile_args else "-O3")
        options = [compile_options]
        if has_data_sets:
            options += ["--verify=ref"]
            options += ["--verify-every-value"]
            options += ["--load-ref={}".format(data_set)]
        if model_name in RunONNXModel_additional_options:
            options += RunONNXModel_additional_options[model_name]
        if args.compile_only:
            options += ["--compile-only"]
        options += ["--model={}".format(onnx_file)]
        if args.log_to_file:
            options += ["--log-to-file={}".format(args.log_to_file)]
        # Wait up to 30 minutes for compilation and inference to finish
        ok, msg = execute_commands(RUN_ONNX_MODEL_CMD + options, tmout=1800)
        state = TEST_PASSED if ok else TEST_FAILED
        logger.info("[{}] check {}".format(model_name, "passed" if ok else "failed"))
        logger.debug("[{}] {}".format(model_name, msg))

        if args.Html:
            with open(os.path.join(report_dir, model_name + ".html"), "w") as out:
                out.write("<html><body><pre>\n")
                out.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                out.write(model_name + "\n\n")
                out.write(msg)
                out.write("</pre></body></html>\n")

    return state


def pull_and_check_model(model_path, compile_args, keep_model, work_dir, report_dir):
    state = TEST_SKIPPED

    # Must get logger again since this function is run by Parallel
    # in a separate process so logger is not propagated.
    logger = get_logger()

    # Ignore deprecated models.
    model_tar_gz = os.path.join(work_dir, model_path.split("/")[-1])
    model_name = model_path.split("/")[-1][: -len(".tar.gz")]  # remove .tar.gz
    if model_name in excluded_models:
        logger.warning("[{}] is excluded. Ignored.".format(model_name))
        return state, model_name

    # pull the model.
    model_url = ONNX_MODEL_ZOO_DOWNLOAD + "/" + model_path
    logger.debug("Downloading {}".format(model_url))
    ok, _ = execute_commands(
        CURL_CMD + [model_url, "--time-cond", model_tar_gz, "--output", model_tar_gz],
        cwd=work_dir,
    )

    # check the model.
    state = check_model(model_tar_gz, model_name, compile_args, report_dir)

    if not keep_model:
        # remove the model to save the storage space.
        os.remove(model_tar_gz)

    return state, model_name


def output_report(
    history_dir,
    report_dir,
    skipped_models,
    tested_models,
    passed_models,
    failed_models,
    total_models,
):
    # Ignore path in args.Html
    html_file = os.path.basename(args.Html)  # foo.html
    json_file = os.path.splitext(html_file)[0] + ".json"  # foo.json
    hist_file = json_file + ".html"  # foo.json.html

    # We used to save the history json in the publish directory but that
    # has problem with concurrent builds. After the publish directory is
    # mounted into the model zoo check container and before we come here
    # to read the json file, another non-merging build could finish and
    # do its publishing (actually just copy and re-publish). This causes
    # the publish directory to be deleted and recreated. So we lost the
    # json file and our history gets reset.
    #
    # So now we save the history json in the job directory so it won't be
    # affected by concurrent builds. Note that reading/writing the json
    # file is not protected since non-merging builds don't touch it. Other
    # merging builds won't be a problem either since only one merging build
    # can run (previously running one gets aborted).
    json_path = os.path.join(history_dir, json_file)

    try:
        with open(json_path, "r") as jf:
            hist = json.load(jf)
        prev = hist[0]
    except:
        hist = []
        prev = {
            "_mesg": "",
            "author": "",
            "commit": "",
            "date": "",
            "failed": {"_models": [], "dropped": [], "entered": []},
            "passed": {"_models": [], "dropped": [], "entered": []},
            "skipped": {"_models": [], "dropped": [], "entered": []},
            "total": {"_models": [], "dropped": [], "entered": []},
        }

    curr = {
        "_mesg": "",
        "author": "",
        "commit": "",
        "date": "",
        "failed": {},
        "passed": {},
        "skipped": {},
        "total": {},
    }

    curr["_mesg"] = os.getenv("ONNX_MLIR_HEAD_COMMIT_MESSAGE", "")
    curr["author"] = os.getenv("ONNX_MLIR_HEAD_COMMIT_AUTHOR", "")
    curr["commit"] = os.getenv("ONNX_MLIR_HEAD_COMMIT_HASH", "")
    curr["date"] = os.getenv("ONNX_MLIR_HEAD_COMMIT_DATE", "")

    curr["failed"]["_models"] = failed_models
    curr["passed"]["_models"] = passed_models
    curr["skipped"]["_models"] = skipped_models
    curr["total"]["_models"] = total_models

    curr["failed"]["dropped"] = [
        x for x in prev["failed"]["_models"] if x not in failed_models
    ]
    curr["passed"]["dropped"] = [
        x for x in prev["passed"]["_models"] if x not in passed_models
    ]
    curr["skipped"]["dropped"] = [
        x for x in prev["skipped"]["_models"] if x not in skipped_models
    ]
    curr["total"]["dropped"] = [
        x for x in prev["total"]["_models"] if x not in total_models
    ]

    curr["failed"]["entered"] = [
        x for x in failed_models if x not in prev["failed"]["_models"]
    ]
    curr["passed"]["entered"] = [
        x for x in passed_models if x not in prev["passed"]["_models"]
    ]
    curr["skipped"]["entered"] = [
        x for x in skipped_models if x not in prev["skipped"]["_models"]
    ]
    curr["total"]["entered"] = [
        x for x in total_models if x not in prev["total"]["_models"]
    ]

    # Write history json. Keep last 100 commits
    HIST_MAX = 100
    hist_json = [curr] + hist[: HIST_MAX - 1]
    with open(json_path, "w") as jf:
        json.dump(hist_json, jf)

    # Write history html
    with open(os.path.join(report_dir, hist_file), "w") as html:
        html.write(
            '<div id="history"></div>\n'
            + "<style>\n"
            + "  .renderjson a { text-decoration: none; }\n"
            + "  .renderjson .disclosure { font-size: 75%; }\n"
            + "</style>\n"
            + '<script type="text/javascript" src="renderjson.js"></script>\n'
            + "<script>\n"
            + '  renderjson.set_icons("\\u{2795}", "\\u{2796}");\n'
            + "  renderjson.set_sort_objects(true);\n"
            + "  renderjson.set_show_to_level(3);\n"
            + '  document.getElementById("history").appendChild(renderjson('
            + json.dumps(hist_json)
            + "));\n"
            + "</script>\n"
        )

    # Write report html
    with open(os.path.join(report_dir, html_file), "w") as html:
        html.write(
            "<html>\n"
            + "<head>\n"
            + "<style>\n"
            + "table, th, td {\n"
            + "  border: 1px solid black;\n"
            + "  border-collapse: collapse;\n"
            + "  padding: 10px;\n"
            + "  vertical-align: top;\n"
            + "}\n"
            + "table.sticky {\n"
            + "  position: -webkit-sticky;\n"
            + "  position: sticky;\n"
            + "  top: 0;\n"
            + "  background-color: #FFF;\n"
            + "}\n"
            + "</style>\n"
            + "</head>\n"
            + "<body>\n"
            + '<table class="sticky">\n'
        )

        t = ["Skipped", "Passed", "Failed"]
        for i, s in enumerate(
            [
                skipped_models,
                list(
                    map(
                        lambda m: (
                            '<a href="'
                            + m
                            + '.html" '
                            + 'target="output">'
                            + m
                            + "</a>"
                        ),
                        passed_models,
                    )
                ),
                list(
                    map(
                        lambda m: (
                            '<a href="'
                            + m
                            + '.html" '
                            + 'target="output">'
                            + m
                            + "</a>"
                        ),
                        failed_models,
                    )
                ),
            ]
        ):
            html.write(
                "  <tr>\n"
                + "    <td>{}</td>\n".format(t[i])
                + "    <td>{}</td>\n".format(len(s))
                + "    <td>{}</td>\n".format(", ".join(s))
                + "  </tr>\n"
            )

        html.write(
            "  <tr>\n"
            + "    <td>Total</td>\n"
            + "    <td>{}</td>\n".format(len(skipped_models) + len(tested_models))
            + '    <td>[ <a href="'
            + hist_file
            + '" target="output">History</a> ]</td>\n'
            + "  </tr>\n"
            + "</table>\n"
            + '<iframe name="output" scrolling="auto"'
            + ' style="border:0px;width:100%;height:100%">\n'
            + "</body>\n"
            + "</html>\n"
        )


def main():
    work_dir = os.path.realpath(args.workdir)
    repo_dir = clone_modelzoo_source(ONNX_MODEL_ZOO_URL, work_dir)
    if not repo_dir:
        logger.error("failed to clone or reset model zoo repo")
        return

    # Collect all model paths in the model zoo
    all_model_names, all_model_paths = obtain_all_model_paths(repo_dir)
    if args.print_paths:
        for path in all_model_paths:
            print(path)
        return

    # By default, run all models in the model zoo.
    # But, if `-m` is specified, the list of models specified are split
    # into models_to_run, e.g.,
    # models_to_run = ['mnist-8', 'yolov4', 'resnet50-v2-7']
    models_to_run = all_model_names

    if args.model:
        models_to_run = args.model.split()

    target_model_paths = set()
    for name in models_to_run:
        if name not in all_model_names:
            logger.error(
                "Model",
                name,
                "not found. Do you mean one of the following? ",
                difflib.get_close_matches(name, all_model_names, len(all_model_names)),
            )
            return
        for m in all_model_paths:
            if name in m:
                target_model_paths.add(m)

    # Start processing the models.
    report_dir = os.path.realpath(args.reportdir)

    results = Parallel(n_jobs=args.jobs, verbose=VERBOSITY_LEVEL[args.log_level])(
        delayed(pull_and_check_model)(
            path, args.compile_args, args.keep_models, work_dir, report_dir
        )
        for path in target_model_paths
    )

    # Report the results.
    skipped_models = sorted(excluded_models)
    tested_models = sorted({r[1] for r in results if r[0] != TEST_SKIPPED})
    passed_models = sorted({r[1] for r in results if r[0] == TEST_PASSED})
    failed_models = sorted({r[1] for r in results if r[0] == TEST_FAILED})
    total_models = sorted(skipped_models + tested_models)

    if args.Html:
        # Output report files
        history_dir = os.path.realpath(args.historydir)
        output_report(
            history_dir,
            report_dir,
            skipped_models,
            tested_models,
            passed_models,
            failed_models,
            total_models,
        )

        # Output summary to stdout for the badge text
        print(
            "Total:{} Skipped:{} Passed:{} Failed:{}".format(
                len(skipped_models) + len(tested_models),
                len(skipped_models),
                len(passed_models),
                len(failed_models),
            )
        )
    else:
        print(
            "{} models tested: {}\n".format(
                len(tested_models), ", ".join(tested_models)
            )
        )
        print(
            "{} models passed: {}\n".format(
                len(passed_models), ", ".join(passed_models)
            )
        )
        print(
            "{} models failed: {}\n".format(
                len(failed_models), ", ".join(failed_models)
            )
        )


if __name__ == "__main__":
    main()
