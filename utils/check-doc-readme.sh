#!/bin/bash

ROOT_DIR=$1
DOC_DIR=$1/docs

README="README.md"

echo "Checking whether docs/README.md is up-to-date or not ..."

diff ${ROOT_DIR}/${README} ${DOC_DIR}/${REAMDE}

if [ $? == 0 ]
then
  # up-to-date.
  echo "docs/README.md is up-to-date."
else
  # not up-to-date.
  echo "docs/README.md is not up-to-date, please copy README.md from the root folder."
  exit 1
fi
