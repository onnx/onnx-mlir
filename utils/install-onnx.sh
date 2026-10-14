#!/usr/bin/env bash
git fetch --prune --unshallow --tags
python3 -m pip install .
rm -rf ${HOME}/.cache
