# SPDX-License-Identifier: Apache-2.0

import pytest


# Add option --silent to pytest
def pytest_addoption(parser):
    parser.addoption("--silent", action="store_true", default=False)


# Function to disable pytest progressive indicators to avoid
# cluttering the output when running in a parallel CI environment.
def pytest_report_teststatus(report, config):
    category, short, verbose = "", "", ""
    if not config.getoption("--silent"):
        return None

    if hasattr(report, "wasxfail"):
        if report.skipped:
            category = "xfailed"
        elif report.passed:
            category = "xpassed"
        return (category, short, verbose)
    elif report.when in ("setup", "teardown"):
        if report.failed:
            category = "error"
        elif report.skipped:
            category = "skipped"
        return (category, short, verbose)
    category = report.outcome
    return (category, short, verbose)
