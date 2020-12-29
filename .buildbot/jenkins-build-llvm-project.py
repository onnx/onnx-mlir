#!/usr/bin/env python3

import datetime
import docker
import hashlib
import json
import logging
import os
import re
import requests
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

READ_CHUNK_SIZE            = 1024*1024

LLVM_PROJECT_SHA1_FILE     = 'utils/clone-mlir.sh'
LLVM_PROJECT_SHA1_REGEX    = 'git checkout ([0-9a-f]+)'
LLVM_PROJECT_DOCKERFILE    = 'docker/Dockerfile.llvm-project'
LLVM_PROJECT_GITHUB_URL    = 'https://api.github.com/repos/llvm/llvm-project'

LLVM_PROJECT_IMAGE         = { 'static': 'onnx-mlir-llvm-static',
                               'shared': 'onnx-mlir-llvm-shared' }
BUILD_SHARED_LIBS          = { 'static': 'off',
                               'shared': 'on' }
LLVM_PROJECT_LABELS        = [ 'llvm_project_sha1',
                               'llvm_project_sha1_date',
                               'llvm_project_dockerfile_sha1' ]

cpu_arch                   = os.getenv('CPU_ARCH')
github_jenkins_droid_token = os.getenv('GITHUB_JENKINS_DROID_TOKEN')
onnx_mlir_pr_number        = os.getenv('ONNX_MLIR_PR_NUMBER')
dockerhub_user_name        = os.getenv('DOCKERHUB_USER_NAME')
docker_daemon_socket       = os.getenv('DOCKER_DAEMON_SOCKET')

docker_api                 = docker.APIClient(base_url=docker_daemon_socket)

# Validate whether the commit sha1 date is a valid UTC ISO 8601 date
def valid_sha1_date(sha1_date):
    try:
        datetime.datetime.strptime(sha1_date, '%Y-%m-%dT%H:%M:%SZ')
        return True
    except:
        return False

# Extract a regex pattern from a file. Used to get llvm-project sha1
# from utils/clone-mlir.sh.
def extract_pattern_from_file(file_name, regex_pattern):
    try:
        for line in open(file_name):
            matched = re.search(re.compile(regex_pattern), line)
            if matched:
                return matched.group(1)
    except:
        return ''

# Get the author commit date of a commit sha
def get_repo_sha1_date(github_repo, commit_sha1):
    try:
        resp = requests.get(
            github_repo + '/commits/' + commit_sha1,
            headers = {
                'Accept': 'application/json',
                'Authorization': 'token ' + github_jenkins_droid_token
            })
        resp.raise_for_status()
        return resp.json()['commit']['committer']['date']
    except:
        logging.info(sys.exc_info()[1])
        return ''

# Compute sha1 of a file
def compute_file_sha1(file_name):
    sha1sum = hashlib.sha1()
    try:
        with open(file_name, 'rb') as f:
            for data in iter(lambda: f.read(READ_CHUNK_SIZE), b''):
                sha1sum.update(data)
        return sha1sum.hexdigest()
    except:
        return ''

# Get the labels of a docker image in the docker registry.
# python docker SDK does not support this so we have to make
# our own REST calls.
def get_remote_image_labels(user_name, image_name, image_tag, image_labels):
    try:
        # Get access token
        resp = requests.get(
            'https://auth.docker.io/token?scope=repository:' +
            user_name + '/' + image_name +
            ':pull&service=registry.docker.io')
        resp.raise_for_status()
        access_token = resp.json()['token']

        # Get manifest, only v1 schema has labels so accept v1 only
        resp = requests.get(
            'https://registry-1.docker.io/v2/' +
            user_name + '/' + image_name + '/manifests/' + image_tag,
            headers={
                'Accept': 'application/vnd.docker.distribution.manifest.v1+json',
                'Authorization': 'Bearer ' + access_token })
        resp.raise_for_status()

        # v1Compatibility is a quoted JSON string, not a JSON object
        manifest = json.loads(resp.json()['history'][0]['v1Compatibility'])
        logging.info('remote %s/%s:%s labels: %s', user_name, image_name, image_tag,
                     manifest['config']['Labels'])
        labels = manifest['config']['Labels']
        if (labels):
            labels_ok = True
            for label in image_labels:
                if not labels[label]:
                    labels_ok = False
                    break
            if labels_ok:
                return labels
        raise Exception('remote image {}/{}:{} does not exist ' +
                        'or has invalid labels'.format(
                            user_name, image_name, image_tag))
    except:
        logging.info(sys.exc_info()[1])
        return ''

# From the pull request source, extract expected llvm-project sha1, sha1 date,
# and dockerfile sha1.
def extract_llvm_info():
    exp_llvm_project_sha1  = extract_pattern_from_file(LLVM_PROJECT_SHA1_FILE,
                                                       LLVM_PROJECT_SHA1_REGEX)
    exp_llvm_project_sha1_date = get_repo_sha1_date(LLVM_PROJECT_GITHUB_URL,
                                                    exp_llvm_project_sha1)
    exp_llvm_project_dockerfile_sha1 = compute_file_sha1(LLVM_PROJECT_DOCKERFILE)

    # Labels used to filter local images
    exp_llvm_project_filter = { 'label': [
        'llvm_project_sha1=' + exp_llvm_project_sha1,
        'llvm_project_dockerfile_sha1=' + exp_llvm_project_dockerfile_sha1 ] }

    logging.info('llvm-project expected')
    logging.info('commit sha1     %s', exp_llvm_project_sha1)
    logging.info('commit date     %s', exp_llvm_project_sha1_date)
    logging.info('dockerfile sha1 %s', exp_llvm_project_dockerfile_sha1)

    return { 'llvm_project_sha1': exp_llvm_project_sha1,
             'llvm_project_sha1_date': exp_llvm_project_sha1_date,
             'llvm_project_dockerfile_sha1': exp_llvm_project_dockerfile_sha1,
             'llvm_project_filter': exp_llvm_project_filter }

# Pull or build llvm-project images, which is required for building our
# onnx-mlir dev and user images. Each pull request will be using its own
# "private" llvm-project images, which have the pull request number as
# the image tag.
def setup_private_llvm(image_type, exp):
    user_name    = dockerhub_user_name
    image_name   = LLVM_PROJECT_IMAGE[image_type]
    image_tag    = onnx_mlir_pr_number
    image_repo   = user_name + '/' + image_name
    image_full   = image_repo + ':' + image_tag
    image_filter = exp['llvm_project_filter']
    image_labels  = LLVM_PROJECT_LABELS

    # First look for a local llvm-project image for the pull request that
    # was built by a previous build job. We can use it if it has both the
    # expected llvm-project sha1 and Dockerfile.llvm-project sha1 (i.e.,
    # the pull request did not modify the Dockerfile.llvm-project that was
    # used to build the llvm-project image.
    id = docker_api.images(name = image_full, filters = image_filter,
                           all = False, quiet = True)

    # If a local useable llvm-project image was not found, see if we can
    # pull one from the registry.
    if not id:
        labels = get_remote_image_labels(
            user_name, image_name, cpu_arch, image_labels)

        # Image in registry has expected llvm-project commit sha1 and
        # Dockerfile.llvm-project sha1, pull and tag it with pull request
        # number for our private use.
        if (labels and
            labels['llvm_project_sha1'] == exp['llvm_project_sha1'] and
            labels['llvm_project_dockerfile_sha1'] == exp['llvm_project_dockerfile_sha1']):
            try:
                for line in docker_api.pull(image_repo, tag = cpu_arch,
                                            stream = True, decode = True):
                    print((line['id']+': '
                           if 'id' in line and 'progress' not in line else '') +
                          (line['status'] + '\n'
                           if 'status' in line and 'progress' not in line else ''),
                          end='', flush=True)

                docker_api.tag(image_repo + ':' + cpu_arch,
                               image_repo, onnx_mlir_pr_number, force=True)

                id = docker_api.images(name = image_full,
                                       all = False, quiet = True)
                logging.info('image %s (%s) pulled', image_full, id[0][0:19])
                return
            except:
                labels['llvm_project_sha1_date'] = ''

        # Build llvm-project locally if one of the following is true
        #
        # - image in registry does not exist
        # - pull image failed
        # - image in registry has an invalid llvm-project commit sha1 date
        #   (should never happen)
        # - expected llvm-project commit sha1 date is invalid (fetch sha1
        #   date failed)
        # - image in registry has an llvm-project commit sha1 date earlier
        #   than what we expect (registry image out of date)
        #
        # Note that if pull failed labels['llvm_project_sha1_date'] will
        # be cleared to make valid_sha1_date false.
        if (not labels or
            not valid_sha1_date(labels['llvm_project_sha1_date']) or
            not valid_sha1_date(exp['llvm_project_sha1_date']) or
            labels['llvm_project_sha1_date'] <= exp['llvm_project_sha1_date']):
            for line in docker_api.build(
                    path = '.',
                    dockerfile = LLVM_PROJECT_DOCKERFILE,
                    tag = image_full,
                    decode = True,
                    rm = True,
                    buildargs = {
                        'BUILD_SHARED_LIBS': BUILD_SHARED_LIBS[image_type],
                        'LLVM_PROJECT_SHA1': exp['llvm_project_sha1'],
                        'LLVM_PROJECT_SHA1_DATE': exp['llvm_project_sha1_date'],
                        'LLVM_PROJECT_DOCKERFILE_SHA1': exp['llvm_project_dockerfile_sha1'],
                        'ONNX_MLIR_PR_NUMBER': onnx_mlir_pr_number
                    }):
                print(line['stream'] if 'stream' in line else '',
                      end='', flush=True)

            id = docker_api.images(name = image_full,
                                   all = False, quiet = True)
            logging.info('image %s (%s) built', image_full, id[0][0:19])

        # Registry image has an llvm-project commit sha1 date later than what
        # we expect, the build source is out of date. Exit to fail the build,
        # regardless of Dockerfile.llvm-project sha1 being expected or not.
        else:
            raise Exception('PR source out of date, rebase then rebuild')

    # Found useable local image
    else:
        logging.info('image %s (%s) found', image_full, id[0][0:19])

def main():
    exp = extract_llvm_info()
    setup_private_llvm('static', exp)
    setup_private_llvm('shared', exp)

if __name__ == "__main__":
    main()
