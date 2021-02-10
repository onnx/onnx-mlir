#!/usr/bin/env python3

import datetime
import docker
import git
import hashlib
import json
import logging
import os
import requests
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

READ_CHUNK_SIZE      = 1024*1024

LLVM_PROJECT_IMAGE   = { 'dev': 'onnx-mlir-llvm-static',
                         'usr': 'onnx-mlir-llvm-shared' }
ONNX_MLIR_IMAGE      = { 'dev': 'onnx-mlir-dev',
                         'usr': 'onnx-mlir' }
ONNX_MLIR_DOCKERFILE = { 'dev': 'docker/Dockerfile.onnx-mlir-dev',
                         'usr': 'docker/Dockerfile.onnx-mlir' }
ONNX_MLIR_LABELS     = [ 'onnx_mlir_sha1',
                         'onnx_mlir_sha1_date',
                         'onnx_mlir_dockerfile_sha1' ]

cpu_arch             = os.getenv('CPU_ARCH')
dockerhub_user_name  = os.getenv('DOCKERHUB_USER_NAME')
docker_daemon_socket = os.getenv('DOCKER_DAEMON_SOCKET')
onnx_mlir_pr_number  = os.getenv('ONNX_MLIR_PR_NUMBER')

docker_api           = docker.APIClient(base_url=docker_daemon_socket)

# Validate whether the commit sha1 date is a valid UTC ISO 8601 date
def valid_sha1_date(sha1_date):
    try:
        datetime.datetime.strptime(sha1_date, '%Y-%m-%dT%H:%M:%SZ')
        return True
    except:
        return False

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

# Get onnx-mlir commit sha1 and date we are expecting to build
# from the local pull request repo.
def get_onnx_mlir_info(image_type, local_repo):
    repo = git.Repo(local_repo)
    exp_onnx_mlir_sha1 = repo.head.commit.hexsha
    exp_onnx_mlir_sha1_date = datetime.datetime.utcfromtimestamp(
        repo.head.commit.committed_date).isoformat() + 'Z'

    exp_onnx_mlir_dockerfile_sha1 = compute_file_sha1(
        ONNX_MLIR_DOCKERFILE[image_type])

    # Labels used to filter local images
    exp_onnx_mlir_filter = { 'label': [
        'onnx_mlir_sha1=' + exp_onnx_mlir_sha1,
        'onnx_mlir_dockerfile_sha1=' + exp_onnx_mlir_dockerfile_sha1 ] }

    logging.info('%s expected', ONNX_MLIR_IMAGE[image_type])
    logging.info('commit sha1:     %s', exp_onnx_mlir_sha1)
    logging.info('commit date:     %s', exp_onnx_mlir_sha1_date)
    logging.info('dockerfile sha1: %s', exp_onnx_mlir_dockerfile_sha1)

    return { 'onnx_mlir_sha1': exp_onnx_mlir_sha1,
             'onnx_mlir_sha1_date': exp_onnx_mlir_sha1_date,
             'onnx_mlir_dockerfile_sha1': exp_onnx_mlir_dockerfile_sha1,
             'onnx_mlir_filter': exp_onnx_mlir_filter }

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

# Build onnx-mlir dev and user images.
def build_private_onnx_mlir(image_type, exp):
    user_name    = dockerhub_user_name
    image_name   = ONNX_MLIR_IMAGE[image_type]
    image_tag    = onnx_mlir_pr_number
    image_repo   = user_name + '/' + image_name
    image_full   = image_repo + ':' + image_tag
    image_filter = exp['onnx_mlir_filter']
    image_labels = ONNX_MLIR_LABELS

    # First look for a local onnx-mlir image for the pull request that
    # was built by a previous build job. We can use it if it has the
    # expected onnx-mlir sha1, which means that onnx-mlir hasn't changed.
    # This is useful for situations where we trigger the build by the
    # "{test|publish} this please" comment phrase for various testing
    # purposes without actually changing the onnx-mlir itself, e.g.,
    # testing different Jenkins job configurations.
    #
    # Note that, unlike the case with llvm-project images,  we don't need
    # to check the dockerfile sha1 used to built the onnx-mlir images.
    # because the dockerfile is part of onnx-mlir. If we changed it, then
    # onnx-mlir commit sha1 would have changed.
    id = docker_api.images(name = image_full, filters = image_filter,
                           all = False, quiet = True)

    # If a local useable onnx-mlir image was not found, see if we can
    # pull one from the registry.
    if not id:
        labels = get_remote_image_labels(
            user_name, image_name, cpu_arch, image_labels)

        # Image in registry has expected onnx-mlir commit sha1, pull and
        # tag it with pull request number for our private use.
        if (labels and
            labels['onnx_mlir_sha1'] == exp['onnx_mlir_sha1']):
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
                logging.info(sys.exc_info()[1])

        # Build onnx-mlir locally if one of the following is true
        #
        # - image in registry does not exist
        # - pull image failed
        # - image in registry has an onnx-mlir commit sha1 different
        #   from what we expect
        #
        for line in docker_api.build(
                path = '.',
                dockerfile = ONNX_MLIR_DOCKERFILE[image_type],
                tag = image_repo + ':' + onnx_mlir_pr_number,
                decode = True,
                rm = True,
                buildargs = {
                    'BASE_IMAGE': dockerhub_user_name + '/' +
                                  LLVM_PROJECT_IMAGE[image_type] + ':' +
                                  onnx_mlir_pr_number,
                    'ONNX_MLIR_SHA1': exp['onnx_mlir_sha1'],
                    'ONNX_MLIR_SHA1_DATE': exp['onnx_mlir_sha1_date'],
                    'ONNX_MLIR_DOCKERFILE_SHA1': exp['onnx_mlir_dockerfile_sha1'],
                    'ONNX_MLIR_PR_NUMBER': onnx_mlir_pr_number
                }):
            print(line['stream'] if 'stream' in line else '',
                  end='', flush=True)

        id = docker_api.images(name = image_full, all = False, quiet = True)
        logging.info('image %s (%s) built', image_full, id[0][0:19])

    # Found useable local image
    else:
        logging.info('image %s (%s) found', image_full, id[0][0:19])

def main():
    build_private_onnx_mlir('dev', get_onnx_mlir_info('dev', '.'))
    build_private_onnx_mlir('usr', get_onnx_mlir_info('usr', '.'))

if __name__ == "__main__":
    main()
