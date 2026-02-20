#!/usr/bin/env bash
# Currently, no real check is done.
# Next PR will perform real check
echo "Start build OMPyInfer..."
if [ "$(uname -m)" = "s390x" ]; then 
  echo "this is s360 machine"
else
  echo "this is NOT s360 machine"
fi
