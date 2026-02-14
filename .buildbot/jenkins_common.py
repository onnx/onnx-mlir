#!/usr/bin/env python3
import datetime
import docker
import fasteners
import git
import hashlib
import jenkins
import logging
import math
import os
import re
import requests
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set parallel jobs based on both CPU count and memory size.
# Because using CPU count alone can result in out of memory
# and get Jenkins killed. For example, we may have 64 CPUs
# (128 threads) and only 32GB memory. So spawning off 128
# cc/c++ processes is going to quickly exhaust the memory.
#
# Algorithm: NPROC = min(2, # of CPUs) if memory < 8GB, otherwise
#            NPROC = min(memory / 8, # of CPUs)
MEMORY_IN_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3)
NPROC = str(math.ceil(min(max(2, MEMORY_IN_GB / 8), os.cpu_count())))

RETRY_LIMIT = 10
READ_CHUNK_SIZE = 1024 * 1024
BASE_BRANCH = "main"

DOCKER_API_TIMEOUT = 3600
DOCKER_DIST_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_DIST_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"

cpu_arch = os.getenv("CPU_ARCH")
docker_pushpull_rwlock = os.getenv("DOCKER_PUSHPULL_RWLOCK")
docker_daemon_socket = os.getenv("DOCKER_DAEMON_SOCKET")
docker_registry_host_name = os.getenv("DOCKER_REGISTRY_HOST_NAME")
docker_registry_user_name = os.getenv("DOCKER_REGISTRY_USER_NAME")
docker_registry_login_name = os.getenv("DOCKER_REGISTRY_LOGIN_NAME")
docker_registry_login_token = os.getenv("DOCKER_REGISTRY_LOGIN_TOKEN")
docker_registry_token_access = os.getenv("DOCKER_REGISTRY_TOKEN_ACCESS")
docker_rwlock = fasteners.InterProcessReaderWriterLock(docker_pushpull_rwlock)
docker_api = docker.APIClient(base_url=docker_daemon_socket, timeout=DOCKER_API_TIMEOUT)

github_repo_access_token = os.getenv("GITHUB_REPO_ACCESS_TOKEN")
github_repo_name = os.getenv("GITHUB_REPO_NAME")
github_repo_name2 = os.getenv("GITHUB_REPO_NAME").replace("-", "_")
github_pr_baseref = os.getenv("GITHUB_PR_BASEREF")
github_pr_baseref2 = os.getenv("GITHUB_PR_BASEREF").lower()
github_pr_number = os.getenv("GITHUB_PR_NUMBER")
github_pr_number2 = os.getenv("GITHUB_PR_NUMBER2")
github_pr_action = os.getenv("GITHUB_PR_ACTION")
github_pr_merged = os.getenv("GITHUB_PR_MERGED")
github_pr_phrase = os.getenv("GITHUB_PR_PHRASE")
github_pr_request_url = os.getenv("GITHUB_PR_REQUEST_URL")
GITHUB_REPO_NAME = github_repo_name.upper()
GITHUB_REPO_NAME2 = github_repo_name2.upper()

jenkins_rest_api_url = os.getenv("JENKINS_REST_API_URL")
jenkins_rest_api_user = os.getenv("JENKINS_REST_API_USER")
jenkins_rest_api_token = os.getenv("JENKINS_REST_API_TOKEN")
jenkins_home = os.getenv("JENKINS_HOME")
jenkins_job_name = os.getenv("JOB_NAME")
jenkins_build_number = os.getenv("BUILD_NUMBER")
jenkins_build_result = os.getenv("JENKINS_BUILD_RESULT")
jenkins_workspace_dir = os.getenv("WORKSPACE")

docker_static_image_name = (
    github_repo_name
    + "-llvm-static"
    + ("." + github_pr_baseref2 if github_pr_baseref != BASE_BRANCH else "")
)
docker_shared_image_name = (
    github_repo_name
    + "-llvm-shared"
    + ("." + github_pr_baseref2 if github_pr_baseref != BASE_BRANCH else "")
)
docker_dev_image_name = (
    github_repo_name
    + "-dev"
    + ("." + github_pr_baseref2 if github_pr_baseref != BASE_BRANCH else "")
)
docker_usr_image_name = github_repo_name + (
    "." + github_pr_baseref2 if github_pr_baseref != BASE_BRANCH else ""
)

pr_image_tag = github_pr_number.lower()

LLVM_PROJECT_LABELS = [
    "llvm_project_sha1",
    "llvm_project_sha1_date",
    "llvm_project_dockerfile_sha1",
]
ONNX_MLIR_LABELS = [
    github_repo_name2 + "_sha1",
    github_repo_name2 + "_sha1_date",
    github_repo_name2 + "_dockerfile_sha1",
]


def resolve_and_override_cpu_arch_from_docker():
    """Detect the Docker daemon/host architecture and, if different from the
    current cpu_arch, override it so we pull/build the correct arch images."""
    global cpu_arch
    try:
        info = docker_api.info()
        arch = info.get("Architecture", "").lower()
        # Map common Docker Architecture values to cpu_arch tag names used by our registry
        arch_map = {
            "x86_64": "amd64",  # Docker daemon reports x86_64, but platform needs amd64
            "amd64": "amd64",
            "aarch64": "arm64",
            "arm64": "arm64",
            "s390x": "s390x",
            "ppc64le": "ppc64le",
        }
        detected = arch_map.get(arch, arch)
        if detected and cpu_arch != detected:
            logging.info(
                "Docker daemon reports Architecture=%s, overriding cpu_arch '%s' -> '%s'",
                arch,
                cpu_arch,
                detected,
            )
            cpu_arch = detected
    except Exception as e:
        logging.warning(
            "Could not detect Docker daemon architecture to resolve cpu_arch: %s", e
        )


def strtobool(s: str) -> bool:
    """Reimplement strtobool per PEP 632 and python 3.12 deprecation."""

    if s.lower() in ["y", "yes", "t", "true", "on", "1"]:
        return True
    elif s.lower() in ["n", "no", "f", "false", "off", "0", ""]:
        return False
    else:
        raise ValueError(f"{s} cannot be converted to bool")


def compute_file_sha1(file_name):
    """Compute sha1 of a file."""

    sha3_256sum = hashlib.sha3_256()
    try:
        with open(file_name, "rb") as f:
            for data in iter(lambda: f.read(READ_CHUNK_SIZE), b""):
                sha3_256sum.update(data)
        return sha3_256sum.hexdigest()
    except:
        return ""


def extract_pattern_from_file(file_name, regex_pattern):
    """Extract a regex pattern from a file."""

    try:
        for line in open(file_name):
            matched = re.search(re.compile(regex_pattern), line)
            if matched:
                return matched.group(1)
    except:
        return ""


def get_repo_sha1_date(github_repo, commit_sha1, access_token):
    """Get the author commit date of a github commit sha."""
    try:
        resp = requests.get(
            url=github_repo + "/commits/" + commit_sha1,
            headers={
                "Accept": "application/json",
                "Authorization": "token " + access_token,
            },
        )
        resp.raise_for_status()
        return resp.json()["commit"]["committer"]["date"]
    except Exception as e:
        logging.exception(e)
        return ""


def valid_sha1_date(sha1_date):
    """Validate whether the commit sha1 date is a valid UTC ISO 8601 date."""

    try:
        datetime.datetime.strptime(sha1_date, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except:
        return False


# Get the labels of a local docker image, raise exception
# if image doesn't exist or has invalid labels.
def get_local_image_labels(host_name, user_name, image_name, image_tag, image_labels):
    image_full = (
        (host_name + "/" if host_name else "")
        + (user_name + "/" if user_name else "")
        + image_name
        + ":"
        + image_tag
    )
    info = docker_api.inspect_image(image_full)
    logging.info("local image %s labels: %s", image_full, info["Config"]["Labels"])
    labels = info["Config"]["Labels"]
    if labels:
        labels_ok = True
        for label in image_labels:
            if not labels[label]:
                labels_ok = False
                break
        if labels_ok:
            return labels
    raise Exception(
        "local image " + image_full + " does not exist or has invalid labels"
    )


# Make REST call to get the access token to operate on an image in
# public docker registry
def get_access_token(host_name, user_name, image_name, login_name, login_token, action):
    resp = requests.get(
        url=(
            "https://"
            + (host_name if host_name else "auth.docker.io")
            + "/token?service=registry.docker.io"
            + "&scope=repository:"
            + (user_name + "/" if user_name else "")
            + image_name
            + ":"
            + action
        ),
        auth=(login_name, login_token),
    )
    resp.raise_for_status()
    return resp.json()["token"]


# Make REST call to get the v1 or v2 manifest of an image from
# public docker registry
#
# ghcr.io only supports v2 manifest which has no v1Compatibility
# so we now get the labels by using v2 config blobs
def get_image_manifest_config(
    host_name,
    user_name,
    image_name,
    image_tag,
    login_name,
    login_token,
    access_token,
):
    # Get manifest
    url = (
        "https://"
        + (host_name if host_name else "registry-1.docker.io")
        + "/v2/"
        + (user_name + "/" if user_name else "")
        + image_name
    )

    headers = {}
    headers["Accept"] = DOCKER_DIST_MANIFEST
    if access_token:
        headers["Authorization"] = "Bearer " + access_token
        auth = None
    else:
        auth = (login_name, login_token)

    manifest = requests.get(
        url=url + "/manifests/" + image_tag, headers=headers, auth=auth
    )
    manifest.raise_for_status()

    # Get config blobs
    config_digest = manifest.json()["config"]["digest"]
    config = requests.get(
        url=url + "/blobs/" + config_digest, headers=headers, auth=auth
    )
    config.raise_for_status()

    return manifest, config


def get_remote_image_labels(
    host_name, user_name, image_name, image_tag, image_labels, login_name, login_token
):
    """Get the labels of a docker image in the docker registry. python docker SDK
    does not support this so we have to make our own REST calls."""

    try:
        access_token = (
            get_access_token(
                host_name, user_name, image_name, login_name, login_token, "pull"
            )
            if strtobool(docker_registry_token_access)
            else None
        )

        _, config = get_image_manifest_config(
            host_name,
            user_name,
            image_name,
            image_tag,
            login_name,
            login_token,
            access_token,
        )

        image_full = (
            (host_name + "/" if host_name else "")
            + (user_name + "/" if user_name else "")
            + image_name
            + ":"
            + image_tag
        )

        labels = config.json()["config"]["Labels"]
        logging.info("remote image %s labels: %s", image_full, labels)
        if labels:
            labels_ok = True
            for label in image_labels:
                if not labels[label]:
                    labels_ok = False
                    break
            if labels_ok:
                return labels
        raise Exception(
            "remote image " + image_full + " does not exist or has invalid labels"
        )
    except Exception as e:
        logging.exception(e)
        return ""


def remove_dependent_containers(image):
    """Remove all the containers depending on an (dangling) image."""

    containers = docker_api.containers(
        filters={"ancestor": image}, all=True, quiet=True
    )
    for container in containers:
        try:
            container_info = docker_api.inspect_container(container["Id"])
            logging.info("Removing     Id:%s", container["Id"])
            logging.info("   Image %s", container_info["Image"])
            logging.info("     Cmd %s", str(container_info["Config"]["Cmd"]))
            logging.info("  Labels %s", str(container_info["Config"]["Labels"]))
            docker_api.remove_container(container["Id"], v=True, force=True)
        except Exception as e:
            logging.exception(e)
            logging.info("errors ignored while removing dependent containers")


def post_pr_comment(url, msg, token):
    """Post a comment on the pull request issue page when the pull request
    source is outdated and publish is rejected."""

    try:
        resp = requests.post(
            url=url,
            headers={"Accept": "application/json", "Authorization": "token " + token},
            data={"body": msg},
        )
        resp.raise_for_status()
        logging.info(
            '{ "url": "%s", "created_at": "%s", '
            + '"updated_at": "%s", "body": "%s" }',
            resp.json()["url"],
            resp.json()["created_at"],
            resp.json()["updated_at"],
            resp.json()["body"],
        )
    except Exception as e:
        logging.exception(e)


def urlretrieve(remote_url, local_file):
    """Download a remote url to local file."""

    req = requests.get(remote_url)
    with open(local_file, "wb") as f:
        f.write(req.content)
