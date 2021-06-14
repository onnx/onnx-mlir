
##
## Test Generator for ONNX-based tests
##

from pathlib import Path
import tempfile
import argparse

import utility

# parse args and set globals
tempdir = Path(tempfile.gettempdir()) / 'MAIA'
parser = argparse.ArgumentParser(description='Generate ONNX test cases.')
parser.add_argument('--outdir', dest='outdir', action='store', default=tempdir,
                    help=f'output directory (default: \"{tempdir}\")')
parser.add_argument('--list', dest='list_only', action='store_true', default=False,
                    help='list tests')
parser.add_argument('--test', dest='test_list', action='append', default=[], metavar="TEST",
                    help='run the listed test (can specify multiple times)')
args = parser.parse_args()

utility.list_only = args.list_only
utility.out_dir = args.outdir
utility.test_list = args.test_list

# if printing the test list, don't print any other output so that tools can easily parse the list
if not utility.list_only:
  print(f"Generating Tests")
  print(f"Models will be placed at \'{utility.out_dir}\':")


# import your test file here -- it should expose a `generate_tests` function
import test_mnist
import test_bert
import test_transpose

# call your `generate_tests` here; it should call `add_test` for each test
test_mnist.generate_tests()
#test_bert.generate_tests()
test_transpose.generate_tests()


# if printing the test list, don't print any other output so that tools can easily parse the list
if not utility.list_only:
  print(f"Test generation complete!")
