#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import platform
import re

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--f', '--file', type=str, required=False, help='input file')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Read in the file
    with open(args.f, 'r') as file:
        filedata = file.read()
        # Replace the target string
        filedata = re.sub('(define [^()]*\([^()]*\))', r'\1 nounwind', filedata)
    
    with open(args.f.replace(".ll", ".nounwind.ll"), 'w') as file:
        file.write(filedata)

if __name__ == "__main__":
    main()