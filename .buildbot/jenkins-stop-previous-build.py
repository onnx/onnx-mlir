#!/usr/bin/env python3

import jenkins
import logging
import os
import time

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S')

JENKINS_STOP_BUILD_TIMEOUT = 60 # seconds

jenkins_rest_api_url   = os.getenv('JENKINS_REST_API_URL')
jenkins_rest_api_user  = os.getenv('JENKINS_REST_API_USER')
jenkins_rest_api_token = os.getenv('JENKINS_REST_API_TOKEN')
jenkins_job_name       = os.getenv('JOB_NAME')
jenkins_build_number   = os.getenv('BUILD_NUMBER')

github_pr_number       = os.getenv('GITHUB_PR_NUMBER')
github_pr_number2      = os.getenv('GITHUB_PR_NUMBER2')

LOG_PULL_PUSH          = ('pull request: #'
                          if github_pr_number == github_pr_number2 else
                          'merge branch: ')

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
    #
    # builds_found has all the old jobs found that should be stopped.
    builds_found = {}
    end_time = time.time() + JENKINS_STOP_BUILD_TIMEOUT
    while time.time() < end_time:
        running_builds = jenkins_server.get_running_builds()
        builds_still_running = {}
        for build in running_builds:
            # Skip ourselves and higher numbered builds. It's possible
            # multiple builds for the same pull request were triggered.
            # Because even though there is a rate limit, the triggered
            # builds might be running very slowly for whatever reason.
            # So a lower numbered build might get to this script first!
            if (build['name'] == job_name and build['number'] >= int(build_number)):
                continue
            build_info = jenkins_server.get_build_info(build['name'],
                                                       build['number'])
            # Each build will have 3 parameters:
            #
            #   - GITHUB_PR_NUMBER_PULL_REQUEST  (when triggered by pull_request)
            #   - GITHUB_PR_NUMBER_ISSUE_COMMENT (when triggered by issue_comment)
            #   - GITHUB_PR_NUMBER_PUSH          (when triggered by push)
            #
            # Only one of them will be set to the correct pull request number
            # and the other two will be set to "none". We simply search for
            # pr_number in all the values in action['parameters'] and stop
            # the previous build if we find the value.
            #
            # Note that all merges will be using the 'main' pull request number
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
                        logging.info('Stopping job %s build #%s for ' +
                                     LOG_PULL_PUSH + '%s',
                                     build['name'], build['number'], pr_number)
                        builds_found[build['number']] = build['name']
                        builds_still_running[build['number']] = build['name']
                        jenkins_server.stop_build(build['name'], build['number'])

        # If we found and tried to stop some old builds, wait 15 seconds before
        # looping back to see if they are gone. Otherwise we are done.
        if builds_still_running:
            time.sleep(15)
        else:
            break

    # After JENKINS_STOP_BUILD_TIMEOUT seconds, we still found old running
    # builds, which means our attempt to stop them failed.
    if builds_still_running:
        raise Exception(('Failed to stop {} for ' + LOG_PULL_PUSH +
                         '{} in {} seconds').format(str(builds_still_running),
                                                    pr_number,
                                                    JENKINS_STOP_BUILD_TIMEOUT))
    # Old running builds found and we successfully stopped all of them
    elif builds_found:
        logging.info('All running builds %s for ' + LOG_PULL_PUSH + '%s stopped',
                     str(builds_found), pr_number)
    # Otherwise, no old running builds found
    else:
        logging.info('No running builds for ' + LOG_PULL_PUSH + '%s found', pr_number)

    logging.info('Runninng job %s build #%s for ' + LOG_PULL_PUSH + '%s',
                 job_name, build_number, pr_number)

def main():
    stop_previous_build(jenkins_job_name, jenkins_build_number, github_pr_number)

if __name__ == "__main__":
    main()
