#!/usr/bin/env python3

import datetime
import docker
import fasteners
import hashlib
import json
import logging
import math
import os
import re
import requests
import sys

logging.basicConfig(
    level = logging.INFO, format = '[%(asctime)s] %(levelname)s: %(message)s')

# Set parallel jobs based on both CPU count and memory size.
# Because using CPU count alone can result in out of memory
# and get Jenkins killed. For example, we may have 64 CPUs
# (128 threads) and only 32GB memory. So spawning off 128
# cc/c++ processes is going to quickly exhaust the memory.
#
# Algorithm: NPROC = min(2, # of CPUs) if memory < 8GB, otherwise
#            NPROC = min(memory / 4, # of CPUs)
MEMORY_IN_GB                = (os.sysconf('SC_PAGE_SIZE') *
                               os.sysconf('SC_PHYS_PAGES') / (1024.**3))
NPROC                       = str(math.ceil(min(max(2, MEMORY_IN_GB/4), os.cpu_count())))

READ_CHUNK_SIZE             = 1024*1024
BASE_BRANCH                 = 'main'

cpu_arch                    = os.getenv('CPU_ARCH')
docker_pushpull_rwlock      = os.getenv('DOCKER_PUSHPULL_RWLOCK')
docker_daemon_socket        = os.getenv('DOCKER_DAEMON_SOCKET')
docker_registry_host_name   = os.getenv('DOCKER_REGISTRY_HOST_NAME')
docker_registry_user_name   = os.getenv('DOCKER_REGISTRY_USER_NAME')
docker_registry_login_name  = os.getenv('DOCKER_REGISTRY_LOGIN_NAME')
docker_registry_login_token = os.getenv('DOCKER_REGISTRY_LOGIN_TOKEN')
github_repo_access_token    = os.getenv('GITHUB_REPO_ACCESS_TOKEN')
github_repo_name            = os.getenv('GITHUB_REPO_NAME')
github_repo_name2           = os.getenv('GITHUB_REPO_NAME').replace('-', '_')
github_pr_baseref           = os.getenv('GITHUB_PR_BASEREF')
github_pr_baseref2          = os.getenv('GITHUB_PR_BASEREF').lower()
github_pr_number            = os.getenv('GITHUB_PR_NUMBER')
github_pr_number2           = os.getenv('GITHUB_PR_NUMBER2')

docker_static_image_name    = (github_repo_name + '-llvm-static' +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != BASE_BRANCH else ''))
docker_shared_image_name    = (github_repo_name + '-llvm-shared' +
                               ('.' + github_pr_baseref2
                                if github_pr_baseref != BASE_BRANCH else ''))

LLVM_PROJECT_SHA1_FILE      = 'utils/clone-mlir.sh'
LLVM_PROJECT_SHA1_REGEX     = 'git checkout ([0-9a-f]+)'
LLVM_PROJECT_DOCKERFILE     = 'docker/Dockerfile.llvm-project'
LLVM_PROJECT_GITHUB_URL     = 'https://api.github.com/repos/llvm/llvm-project'
LLVM_PROJECT_IMAGE          = { 'static': docker_static_image_name,
                                'shared': docker_shared_image_name }
BUILD_SHARED_LIBS           = { 'static': 'off',
                                'shared': 'on' }
LLVM_PROJECT_LABELS         = [ 'llvm_project_sha1',
                                'llvm_project_sha1_date',
                                'llvm_project_dockerfile_sha1' ]

GITHUB_REPO_NAME            = github_repo_name.upper()
GITHUB_REPO_NAME2           = github_repo_name2.upper()

DOCKER_DIST_MANIFESTS       = {
    'v1': 'application/vnd.docker.distribution.manifest.v1+json',
    'v2': 'application/vnd.docker.distribution.manifest.v2+json' }

docker_rwlock               = fasteners.InterProcessReaderWriterLock(docker_pushpull_rwlock)
docker_api                  = docker.APIClient(base_url=docker_daemon_socket)

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
def get_repo_sha1_date(github_repo, commit_sha1, access_token):
    try:
        resp = requests.get(
            url = github_repo + '/commits/' + commit_sha1,
            headers = { 'Accept': 'application/json',
                        'Authorization': 'token ' + access_token
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

# Make REST call to get the v1 or v2 manifest of an image from
# private docker registry
def get_image_manifest_private(host_name, user_name, image_name, image_tag,
                               schema_version, login_name, login_token):
    resp = requests.get(
        url = 'https://' + host_name + '/v2/' +
              (user_name + '/' if user_name else '') +
              image_name + '/manifests/' + image_tag,
        headers = { 'Accept': DOCKER_DIST_MANIFESTS[schema_version] },
        auth = (login_name, login_token))
    resp.raise_for_status()
    return resp

# Make REST call to get the access token to operate on an image in
# public docker registry
def get_access_token(user_name, image_name, action, login_name, login_token):
    resp = requests.get(
        url = 'https://auth.docker.io/token' +
              '?service=registry.docker.io' +
              '&scope=repository:' +
              (user_name + '/' if user_name else '') + image_name + ':'+ action,
        auth = (login_name, login_token))
    resp.raise_for_status()
    return resp.json()['token']

# Make REST call to get the v1 or v2 manifest of an image from
# public docker registry
def get_image_manifest_public(user_name, image_name, image_tag,
                              schema_version, login_name, login_token, access_token = None):
    # Get access token if not passed in
    if not access_token:
        access_token = get_access_token(
            user_name, image_name, 'pull', login_name, login_token)
    # Get manifest
    resp = requests.get(
        url = ('https://registry-1.docker.io/v2/' +
               (user_name + '/' if user_name else '') +
               image_name + '/manifests/' + image_tag),
        headers={ 'Accept': DOCKER_DIST_MANIFESTS[schema_version],
                  'Authorization': 'Bearer ' + access_token })
    resp.raise_for_status()
    return resp

# Get the labels of a docker image in the docker registry.
# python docker SDK does not support this so we have to make
# our own REST calls.
def get_remote_image_labels(host_name, user_name, image_name, image_tag,
                            image_labels, login_name, login_token):
    try:
        # Get manifest, only v1 schema has labels so accept v1 only
        resp = (
            # private docker registry
            get_image_manifest_private(host_name, user_name, image_name, image_tag,
                                       'v1', login_name, login_token)
            if host_name else
            # public docker registry
            get_image_manifest_public(user_name, image_name, image_tag,
                                      'v1', login_name, login_token))

        image_full = ((host_name + '/' if host_name else '') +
                      (user_name + '/' if user_name else '') +
                      image_name + ':' + image_tag)

        # v1Compatibility is a quoted JSON string, not a JSON object
        manifest = json.loads(resp.json()['history'][0]['v1Compatibility'])
        logging.info('remote image %s labels: %s',
                     image_full, manifest['config']['Labels'])
        labels = manifest['config']['Labels']
        if (labels):
            labels_ok = True
            for label in image_labels:
                if not labels[label]:
                    labels_ok = False
                    break
            if labels_ok:
                return labels
        raise Exception('remote image ' + image_full +
                        ' does not exist or has invalid labels')
    except:
        logging.info(sys.exc_info()[1])
        return ''

# From the pull request source, extract expected llvm-project sha1, sha1 date,
# and dockerfile sha1.
def extract_llvm_info():
    exp_llvm_project_sha1  = extract_pattern_from_file(LLVM_PROJECT_SHA1_FILE,
                                                       LLVM_PROJECT_SHA1_REGEX)
    exp_llvm_project_sha1_date = get_repo_sha1_date(LLVM_PROJECT_GITHUB_URL,
                                                    exp_llvm_project_sha1,
                                                    github_repo_access_token)
    exp_llvm_project_dockerfile_sha1 = compute_file_sha1(LLVM_PROJECT_DOCKERFILE)

    # Labels used to filter local images
    exp_llvm_project_filter = { 'label': [
        'llvm_project_sha1=' + exp_llvm_project_sha1,
        'llvm_project_dockerfile_sha1=' + exp_llvm_project_dockerfile_sha1,
        'llvm_project_successfully_built=yes' ] }

    logging.info('llvm-project expected')
    logging.info('commit sha1:     %s', exp_llvm_project_sha1)
    logging.info('commit date:     %s', exp_llvm_project_sha1_date)
    logging.info('dockerfile sha1: %s', exp_llvm_project_dockerfile_sha1)
    logging.info('image filter:    %s', exp_llvm_project_filter)

    return { 'llvm_project_sha1': exp_llvm_project_sha1,
             'llvm_project_sha1_date': exp_llvm_project_sha1_date,
             'llvm_project_dockerfile_sha1': exp_llvm_project_dockerfile_sha1,
             'llvm_project_filter': exp_llvm_project_filter }

# Remove all the containers depending on an (dangling) image.
def remove_dependent_containers(image):
    containers = docker_api.containers(
        filters = { 'ancestor': image }, all=True, quiet=True)
    for container in containers:
        try:
            container_info = docker_api.inspect_container(container['Id'])
            logging.info('Removing     Id:%s', container['Id'])
            logging.info('   Image %s', container_info['Image'])
            logging.info('     Cmd %s', str(container_info['Config']['Cmd']))
            logging.info('  Labels %s', str(container_info['Config']['Labels']))
            docker_api.remove_container(container['Id'], v=True, force=True)
        except:
            logging.info(sys.exc_info()[1])
            logging.info('errors ignored while removing dependent containers')

# Pull or build llvm-project images, which is required for building our
# onnx-mlir dev and user images. Each pull request will be using its own
# "private" llvm-project images, which have the pull request number as
# the image tag.
def setup_private_llvm(image_type, exp):
    host_name    = docker_registry_host_name
    user_name    = docker_registry_user_name
    login_name   = docker_registry_login_name
    login_token  = docker_registry_login_token
    image_name   = LLVM_PROJECT_IMAGE[image_type]
    image_tag    = github_pr_number.lower()
    image_repo   = ((host_name + '/' if host_name else '') +
                    (user_name + '/' if user_name else '') +
                    image_name)
    image_full   = image_repo + ':' + image_tag
    image_arch   = image_repo + ':' + cpu_arch
    image_filter = exp['llvm_project_filter']
    image_labels = LLVM_PROJECT_LABELS

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
        # Acquire read lock to pull the arch image. This is to serialize
        # against other PR merges trying to push (writers) the arch image.
        # PR builds trying to pull (readers) the arch image can go concurrently.
        logging.info('acquiring read lock for pulling %s', image_arch)
        docker_rwlock.acquire_read_lock()
        try:
            labels = get_remote_image_labels(host_name, user_name, image_name, cpu_arch,
                                             image_labels, login_name, login_token)

            # Image in registry has expected llvm-project commit sha1 and
            # Dockerfile.llvm-project sha1, pull and tag it with pull request
            # number for our private use.
            if (labels and
                labels['llvm_project_sha1'] == exp['llvm_project_sha1'] and
                labels['llvm_project_dockerfile_sha1'] == exp['llvm_project_dockerfile_sha1']):

                for line in docker_api.pull(image_repo, tag = cpu_arch,
                                            stream = True, decode = True):
                    print((line['id']+': '
                           if 'id' in line and 'progress' not in line else '') +
                          (line['status'] + '\n'
                           if 'status' in line and 'progress' not in line else ''),
                          end='', flush=True)

                # Tag pulled arch image with pull request number then remove
                # the arch image
                docker_api.tag(image_arch, image_repo, image_tag, force = True)
                docker_api.remove_image(image_arch, force = True)

                # For logging purpose only
                id = docker_api.images(name = image_full,
                                       all = False, quiet = True)
                logging.info('image %s (%s) tagged', image_full, id[0][0:19])
                return
        except:
            labels['llvm_project_sha1_date'] = ''
        # Remove arch image and release lock regardless of exception or not
        finally:
            docker_rwlock.release_read_lock()
            logging.info('released read lock for pulling %s', image_arch)

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
            layer_sha256 = ''
            for line in docker_api.build(
                    path = '.',
                    dockerfile = LLVM_PROJECT_DOCKERFILE,
                    tag = image_full,
                    decode = True,
                    rm = True,
                    buildargs = {
                        'NPROC': NPROC,
                        'BUILD_SHARED_LIBS': BUILD_SHARED_LIBS[image_type],
                        'LLVM_PROJECT_SHA1': exp['llvm_project_sha1'],
                        'LLVM_PROJECT_SHA1_DATE': exp['llvm_project_sha1_date'],
                        'LLVM_PROJECT_DOCKERFILE_SHA1': exp['llvm_project_dockerfile_sha1'],
                        GITHUB_REPO_NAME2 + '_PR_NUMBER': github_pr_number,
                        GITHUB_REPO_NAME2 + '_PR_NUMBER2': github_pr_number2
                    }):

                if 'stream' in line:
                    # Keep track of the latest successful image layer
                    m = re.match('^\s*---> ([0-9a-f]+)$', line['stream'])
                    if m:
                        layer_sha256 = m.group(1)
                    print(line['stream'], end='', flush=True)

                if 'error' in line:
                    # Tag the latest successful image layer for easier debugging.
                    #
                    # It's OK to tag the broken image since it will not have the
                    # llvm_project_successfully_built=yes label so it will not be
                    # incorrectly reused.
                    if layer_sha256:
                        image_layer = 'sha256:' + layer_sha256
                        remove_dependent_containers(image_layer)
                        logging.info('tagging %s -> %s for debugging', image_layer, image_full)
                        docker_api.tag(image_layer, image_repo, image_tag, force=True)
                    else:
                        logging.info('no successful image layer for tagging')
                    raise Exception(line['error'])

            id = docker_api.images(name=image_full, all=False, quiet=True)
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
