#!/usr/bin/env python3

import jenkins
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

JENKINS_STOP_BUILD_TIMEOUT = 60 # seconds

jenkins_rest_api_url   = os.getenv('JENKINS_REST_API_URL')
jenkins_rest_api_user  = os.getenv('JENKINS_REST_API_USER')
jenkins_rest_api_token = os.getenv('JENKINS_REST_API_TOKEN')
jenkins_job_name       = os.getenv('JOB_NAME')
jenkins_build_number   = os.getenv('BUILD_NUMBER')

onnx_mlir_pr_number    = os.getenv('ONNX_MLIR_PR_NUMBER')

# We allow concurrent builds for different pull request numbers
# but for the same pull request number only one build can run. So
# when a new build starts, a previous build with the same pull
# request number needs to be stopped.
#
# One complication is that a pull request may be built by different
# jobs. For example, it can be built by a push triggered job, or by
# a comment triggered job. And there is no correlation between the
# build numbers of the two jobs.
#
# So we simple look for any running build with the same pull request
# number, which is set by the job parameter.
def stop_previous_build(job_name, build_number, pr_number):
    jenkins_server = jenkins.Jenkins(url = jenkins_rest_api_url,
                                     username = jenkins_rest_api_user,
                                     password = jenkins_rest_api_token)
    # If we find and stop a previous build, loop for up to
    # JENKINS_STOP_BUILD_TIMEOUT seconds for it to abort.
    prev_job_name = ''
    prev_build_number = 0
    end_time = time.time() + JENKINS_STOP_BUILD_TIMEOUT
    while time.time() < end_time:
        running_builds = jenkins_server.get_running_builds()
        stopping = False
        for build in running_builds:
            # Skip ourselves
            if (build['name'] == job_name and build['number'] == int(build_number)):
                continue
            build_info = jenkins_server.get_build_info(build['name'],
                                                       build['number'])
            # Each build will have 3 parameters:
            #
            #   - ONNX_MLIR_PR_NUMBER_PULL_REQUEST  (when triggered by pull_request)
            #   - ONNX_MLIR_PR_NUMBER_ISSUE_COMMENT (when triggered by issue_comment)
            #   - ONNX_MLIR_PR_NUMBER_PUSH          (when triggered by push)
            #
            # Only one of them will be set to the correct pull request number
            # and the other two will be set to "none". We simply search for
            # pr_number in all the values in action['parameters'] and stop
            # the previous build if we find the value.
            #
            # Note that all merges will be using the 'master' pull request number
            # so there can be only one merge at a time, across all pull requests.

            for action in build_info['actions']:
                if ('_class' not in action or
                    action['_class'] != 'hudson.model.ParametersAction' or
                    'parameters' not in action or
                    not action['parameters']):
                    continue

                for parameter in action['parameters']:
                    if ('_class' not in parameter or
                        parameter['_class'] != 'hudson.model.StringParameterValue'):
                        continue

                    if 'value' in parameter and parameter['value'] == pr_number:
                        logging.info('Stopping job %s build #%s for pull request #%s',
                                     build['name'], build['number'], pr_number)
                        stopping = True
                        prev_job_name = build['name']
                        prev_build_number = build['number']
                        jenkins_server.stop_build(build['name'], build['number'])

                        # Only one previous build can be running so stop looping if
                        # we found one
                        break
                if stopping:
                    break
            if stopping:
                break

        # If we stopped a previous build, wait 15 seconds before
        # looping back to see if it's gone. Otherwise we are done.
        if stopping:
            time.sleep(15)
        else:
            break

    if stopping:
        raise Exception(('Failed to stop job {} build {} ' +
                         'for pull request #{} in {} seconds').format(
                             prev_job_name, prev_build_number, pr_number,
                             JENKINS_STOP_BUILD_TIMEOUT))

    logging.info('Runninng job %s build #%s for pull request #%s',
                 job_name, build_number, pr_number)

def main():
    stop_previous_build(jenkins_job_name, jenkins_build_number, onnx_mlir_pr_number)

if __name__ == "__main__":
    main()
