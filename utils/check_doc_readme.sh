#!/bin/bash

ROOT_DIR=$1
DOC_DIR=$1/docs

README="README.md"

echo "Checking whether docs/README.md is up-to-date or not ..."

diff ${ROOT_DIR}/${README} ${DOC_DIR}/${REAMDE} > readme.patch

# not up-to-date.
if [ -s readme.patch ]; then
  cat readme.patch
  echo "docs/README.md is not up-to-date, please copy README.md from the root folder."
  rm readme.patch
  exit 1
fi

# up-to-date.
echo "Done."
rm readme.patch
