#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Ground a lit test in actual numbers: compare two lowering variants of the
# same computation and verify that they produce identical (within tolerance)
# numerical results.
#
#   GroundLitTest.py [options] <lit-test-filename>
#
# The file under test is the one positional argument, spelled as fixLitTest.py
# spells it: every run needs it, so a flag would only be a word to type.
#
# The "baseline" can be specified in one of two ways:
#   - A separate .mlir file containing a function with the same name as the
#     one being tested (file mode: -b, or the default
#     "<test file without extension>.baseline<ext>" next to it).
#   - The same test file, compiled with different onnx-mlir options
#     (flag mode: -r/-t/-a), mirroring CheckONNXModel.py's options. Note
#     these are onnx-mlir options, NOT onnx-mlir-opt options -- a lit test's
#     own "// RUN: onnx-mlir-opt ..." line is a different flag namespace and
#     cannot be reused directly here.
#
# In file mode the two isolated functions must genuinely differ: comments are
# stripped from both before they are compared, and two variants that hold the
# same MLIR modulo comments are reported as a FAILURE (nothing was really
# compared, so a "PASS" would be meaningless). That check runs regardless of
# -d/--diff; -d only adds the visible, comment-free side-by-side listing.
#
# -c/--compile-args applies in both modes, to every compile in the run:
#   - File mode: the same options are used for both the baseline and test
#     file (there is no ref/test split to layer onto).
#   - Flag mode: -c is the shared prefix both branches build on, so you don't
#     have to repeat a large common option string in both -r and -t:
#       ref  = c + r
#       test = c + t            (if -t given -- independent of r, same base)
#            = ref + a          (if -a given -- a delta on top of ref itself)
#            = c                (if neither given -- t defaults to empty, same
#                                 as r does when -r is absent)
#
# File mode is for features (like krnl.collapse) where the difference between
# variants cannot be expressed as a single onnx-mlir flag, so the
# CheckONNXModel.py "same file, two option sets" model does not apply -- the
# two variants are genuinely different KRNL/MLIR source. For features that
# *can* be toggled with a flag, flag mode here is equivalent to
# CheckONNXModel.py, just scoped to a single isolated function.
#
# -f/--func picks one function. Without it, every function in the test file is
# tested, one at a time, each reporting what a single -f run reports (bar the
# "reproduce this manually" recipe, which only a single -f run can hand out --
# see below), followed by a summary of which functions succeeded and which
# failed, in the order they appear in the file.
#
# A test file can carry the options it needs to be run with, so that a bare
# "GroundLitTest.py <file>" is enough and nobody has to rediscover, months
# later, that one function only means anything at --shape-info 0:10x20:
#
#   // GROUND-ALL: <options>    in the header, before the first function:
#                              defaults for every function in the file.
#   // GROUND-THIS: <options>   anywhere before a function: defaults for that
#                              one function.
#
# Both take this tool's own options, in either the "--flag" or the single-dash
# "-flag" spelling the .mlir file's own RUN line uses, and both may be repeated
# (a long option list can be spread over several lines). Precedence is per
# option, most specific winning:
#
#   command line   >   GROUND-THIS   >   GROUND-ALL   >   built-in default
#
# so a GROUND-THIS that sets only --shape-info still inherits GROUND-ALL's
# --compile-args, and anything typed on the command line still wins over both.
# A directive cannot name a file, nor set -f/--func (a file naming itself, or
# choosing which of its functions gets tested, is not something it should
# decide). An option value that starts with "-" needs the "--flag=value" form,
# just as it does on the command line. Only the test file is scanned for
# directives; a baseline file's own directives are ignored.
#
# Relies on:
#   - fixLitTest.py -m <func-name> <file>: isolate a single function into a
#     standalone, runnable module (wrapped in "module {...}" with an
#     "onnx.EntryPoint"). RunONNXModel.py specifically requires this
#     "onnx.EntryPoint" marker in the source file before it will compile it.
#   - RunONNXModel.py: compile, run, save/load reference inputs+outputs, and
#     verify against them.

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
FIX_LIT_TEST = os.path.join(UTILS_DIR, "fixLitTest.py")
RUN_ONNX_MODEL = os.path.join(UTILS_DIR, "RunONNXModel.py")
# Bare names for the "reproduce this manually" recipe -- the user is expected
# to know these tools live in utils/, so the full path is just noise there.
FIX_LIT_TEST_NAME = os.path.basename(FIX_LIT_TEST)
RUN_ONNX_MODEL_NAME = os.path.basename(RUN_ONNX_MODEL)

# Canonical, fixed names (like fixLitTest.py's own "flt_*.mlir" files), written
# to the current directory and overwritten on every run -- so there is always
# exactly one copy to inspect after the fact, not one per invocation. Left behind
# only by a single-function (-f) run, whose one copy is worth inspecting; a run
# over every function would leave nothing but its last function's leftovers, so
# it clears them instead.
GLT_TEST_FILE = "glt_test.mlir"
GLT_BASELINE_FILE = "glt_baseline.mlir"
GLT_REF_DIR = "glt_ref"

# Tag that turns a model's name into its file-mode baseline's: "foo.mlir" pairs
# with "foo.baseline.mlir". A dot-delimited segment rather than a "-baseline"
# one, because test/mlir names are kebab-case and already hold variant pairs
# like "add-exec-cpu.mlir"/"add-exec-cpu-opt.mlir", where one more hyphenated
# segment would be indistinguishable from part of a test's own name. A dot
# segment reads as "companion of foo.mlir", maps back to it unambiguously, and
# is safe to match with a glob. test/mlir/lit.cfg.py skips this suffix when
# collecting tests, so a baseline needs no RUN line of its own -- the two must
# be kept in step.
BASELINE_TAG = ".baseline"

# The same notion of "a function" as fixLitTest.py, so that testing all
# functions covers exactly the set fixLitTest.py can isolate.
FUNC_RE = re.compile(r"\s*func.*@(\w+)\(")

# In-file option directives. A "GROUND-THIS" applies to the next function
# defined after it; a "GROUND-ALL" applies to the whole file and so has to sit
# in the header, before any function, where a reader looks for file-wide
# statements. The two names quantify over the same thing -- all the functions,
# or this one -- so that reading either tells you the scope of the other.
GROUND_ALL = "GROUND-ALL"
GROUND_ONE = "GROUND-THIS"
GROUND_ALL_RE = re.compile(rf"\s*//\s*{GROUND_ALL}:(.*)$")
GROUND_ONE_RE = re.compile(rf"\s*//\s*{GROUND_ONE}:(.*)$")
COMMAND_LINE = "command line"

# Where each option can come from, in increasing order of precedence, and the
# defaults everything falls back to. Options are tracked as {dest: value} dicts
# holding only what was actually specified, so that "not given here" and "given
# here as the same value the default happens to have" stay distinguishable and
# the layers below still show through.
DEFAULTS = {
    "model": None,
    "func": None,
    "baseline_model": None,
    "compile_args": None,
    "ref_compile_args": None,
    "test_compile_args": None,
    "additional_test_compile_args": None,
    "diff": False,
    "rtol": "0.05",
    "atol": "0.01",
    "seed": None,
    "shape_info": None,
    "lower_bound": None,
    "upper_bound": None,
    "input_value": None,
    "verbose": False,
}
# A file may not name itself, nor pick which of its functions gets tested.
COMMAND_LINE_ONLY = ("model", "func")


class SetupError(Exception):
    """
    A problem that prevents a function from being compared at all (it cannot be
    isolated, it is missing from the baseline, the baseline itself won't build,
    its options contradict each other, ...) as opposed to a genuine numerical
    mismatch. Fatal when a single function was named with -f; only that one
    function's failure when every function is being tested -- which matters
    here, since each function can bring its own GROUND-THIS options and so its
    own way of being misconfigured.
    """


class DirectiveError(Exception):
    """
    Raised instead of argparse's usual exit-with-usage when the options inside a
    GROUND-ALL/GROUND-THIS directive don't parse, so that the message can name
    the file and line the directive came from rather than a command line the
    user never typed.
    """


class DirectiveParser(argparse.ArgumentParser):
    def error(self, message):
        raise DirectiveError(message)


class Logger:
    """
    All of one comparison's subprocess commands and their output go to a
    private, uniquely-named temp file (not a fixed shared name -- multiple
    users on the same machine could otherwise collide). Shown live only with
    -v; otherwise dumped in full only if that comparison fails, then always
    removed.
    """

    def __init__(self, verbose):
        self.verbose = verbose
        fd, self.path = tempfile.mkstemp(prefix="glt_run_", suffix=".log")
        self.fh = os.fdopen(fd, "w")

    def log(self, text="", end="\n"):
        self.fh.write(text + end)
        self.fh.flush()
        if self.verbose:
            print(text, end=end)

    def dump_on_failure(self):
        print("--- full log ---")
        self.fh.flush()
        with open(self.path) as f:
            sys.stdout.write(f.read())
        print("--- end log ---")

    def close(self):
        self.fh.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


def build_parser(cls=argparse.ArgumentParser, model_required=True):
    """
    One definition of the options, used both for the command line and for the
    in-file directives -- so a directive accepts exactly what can be typed, and
    neither can drift from the other. Defaults live in DEFAULTS rather than in
    the actions: with argument_default=SUPPRESS, an option missing from the
    parsed namespace means "not specified at this level", which is what the
    command-line-over-GROUND-THIS-over-GROUND-ALL layering needs to know.
    """
    parser = cls(
        prog=os.path.basename(__file__),
        argument_default=argparse.SUPPRESS,
        # Wrapped by hand: RawDescriptionHelpFormatter (needed for the epilog's
        # option arithmetic to keep its shape) leaves this text exactly as given.
        description=(
            "Compare two lowering variants of the same MLIR function: "
            "compile and run\nboth, feed them identical inputs, and verify "
            "their outputs match within\ntolerance."
        ),
        epilog=(
            "Two mutually exclusive ways to specify the baseline:\n"
            "  file mode (-b):     a different .mlir file, same function name.\n"
            "  flag mode (-r/-t/-a): the SAME test file, compiled twice with\n"
            "                       different onnx-mlir options.\n"
            "-c/--compile-args applies in both modes, to every compile in the\n"
            "run. In flag mode it is the shared prefix -r/-t/-a build on:\n"
            "  ref  = c + r\n"
            "  test = c + t        (if -t given)\n"
            "       = ref + a      (if -a given)\n"
            "       = c            (if neither given -- t defaults to empty)\n"
            "Compatibility: -b excludes -r/-t/-a. -c combines with all of\n"
            "-b/-r/-t/-a. -t and -a are mutually exclusive with each other.\n"
            "Without -f/--func, every function of the test file is tested in\n"
            "turn.\n"
            "\n"
            "The test file may also carry its own options, as comments holding\n"
            "these same flags:\n"
            f"  // {GROUND_ALL}: <options>   in the header, before the first\n"
            "                              function: defaults for the file.\n"
            f"  // {GROUND_ONE}: <options>  before one function: defaults for\n"
            "                              that function only.\n"
            "Both may be repeated, and neither may name a file or set -f.\n"
            "Precedence, per option:\n"
            f"  {COMMAND_LINE} > {GROUND_ONE} > {GROUND_ALL} > built-in default"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # -h is added by hand below, so that it lands in the first group with
        # everything else this script owns rather than in a section of its own.
        add_help=False,
    )
    # Two groups, so that reading the help does not mean sorting out, flag by
    # flag, which ones are this tool's own doing and which are just handed
    # through to RunONNXModel.py and mean there exactly what they mean there.
    own = parser.add_argument_group(f"{parser.prog}'s own options")
    forwarded = parser.add_argument_group(
        "options forwarded to RunONNXModel.py",
        "Same spelling and meaning as there; this script only passes them on.",
    )

    own.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit."
    )
    # Positional, as in fixLitTest.py: every run needs it, so there is nothing
    # for a flag to distinguish. "?" for the directive parser, which must be able
    # to parse an option-only line -- and, having the argument declared, can then
    # say what is wrong with a directive that does name a file.
    own.add_argument(
        "model",
        nargs=None if model_required else "?",
        metavar="lit-test-filename",
        help="Path to the test .mlir file.",
    )
    own.add_argument(
        "-f",
        "--func",
        help=(
            "Name of the function to isolate and compare. Default: test every "
            "function of the test file, one at a time."
        ),
    )

    own.add_argument(
        "-b",
        "--baseline-model",
        help=(
            "File mode: path to a baseline .mlir file containing a function "
            "with the same name as --func. Default: "
            '"<test file without extension>.baseline<ext>", next to it. '
            "Mutually exclusive with -r/-t/-a. See epilog for the full "
            "compatibility rules."
        ),
    )
    own.add_argument(
        "-c",
        "--compile-args",
        help=(
            "onnx-mlir options applied to EVERY compile in this run, in "
            "either mode. In file mode: the same options for both the "
            "baseline and test file. In flag mode: the shared prefix -r/-t/-a "
            "build on (see epilog). Default: empty."
        ),
    )
    own.add_argument(
        "-r",
        "--ref-compile-args",
        help=(
            "Flag mode: compile the SAME test file twice instead of using a "
            "second file. These are the reference/baseline onnx-mlir "
            "options, appended after -c's (NOT onnx-mlir-opt options). "
            "Default: empty."
        ),
    )
    own.add_argument(
        "-t",
        "--test-compile-args",
        help=(
            "Flag mode: test onnx-mlir options, appended after -c's -- "
            "independent of -r's options, not built on top of them. Use "
            "either -t or -a, not both."
        ),
    )
    own.add_argument(
        "-a",
        "--additional-test-compile-args",
        help=(
            "Flag mode: test onnx-mlir options, added on top of the full "
            "reference options (-c and -r together). Use either -t or -a, "
            "not both."
        ),
    )

    own.add_argument(
        "-d",
        "--diff",
        action="store_true",
        help=(
            "Show a side-by-side diff of the two isolated function bodies "
            "with all comments stripped (file mode), or of the two "
            "compile-arg strings being compared (flag mode)."
        ),
    )
    forwarded.add_argument(
        "--rtol",
        help=("Relative tolerance. Default: " f"{DEFAULTS['rtol']}."),
    )
    forwarded.add_argument(
        "--atol",
        help=("Absolute tolerance. Default: " f"{DEFAULTS['atol']}."),
    )
    forwarded.add_argument(
        "--seed",
        help="Seed for random input generation, for the baseline run.",
    )
    forwarded.add_argument("--shape-info", help="Dynamic input shapes.")
    forwarded.add_argument("--lower-bound", help="Lower bound for random inputs.")
    forwarded.add_argument("--upper-bound", help="Upper bound for random inputs.")
    forwarded.add_argument("--input-value", help="Per-input data fill spec.")
    own.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Show every command and its full output live. Without this, "
            "that output is captured to a private temp file and only printed "
            "(then removed) if the comparison fails."
        ),
    )
    return parser


def flag_table(parser):
    """
    dest -> (flag spelling, does it take a value), for printing an option set
    back as something that could be typed or pasted into a directive. argparse
    exposes no public accessor for the arguments a parser was given, and the
    alternative is a hand-kept second copy of every flag spelling here, free to
    drift from the real one.

    The model has no option strings, being positional, and so falls out of this
    table -- which is what the option listings want anyway: it is reported as the
    file being tested, not as one of the options applied to it.
    """
    table = {}
    for action in parser._actions:
        if action.option_strings and action.dest != "help":
            # The last spelling is the long one ("--func" of "-f/--func").
            table[action.dest] = (action.option_strings[-1], action.nargs != 0)
    return table


CLI_PARSER = build_parser()
DIRECTIVE_PARSER = build_parser(cls=DirectiveParser, model_required=False)
FLAG_BY_DEST = flag_table(CLI_PARSER)
KNOWN_FLAGS = {flag for action in CLI_PARSER._actions for flag in action.option_strings}


def command_line_only_complaint(dest):
    """
    What a directive that reaches for a COMMAND_LINE_ONLY option is doing, named
    the way its author would name it. The model is positional and so has no flag
    to quote back -- and "names a file" says more about the mistake than any
    spelling of it would, since what such a directive holds is a bare path.
    """
    if dest == "model":
        return "names a file"
    return f"sets {FLAG_BY_DEST[dest][0]}"


def normalize_flags(tokens):
    """
    Accept the single-dash spelling of a long option ("-shape-info=0:10x20"),
    which is how every flag in an .mlir file's own RUN line is written and so
    how a directive next to them will be written too. argparse only knows the
    "--shape-info" form, so the missing dash is put back before it looks.
    """
    normalized = []
    for token in tokens:
        name = token.split("=", 1)[0]
        if (
            name.startswith("-")
            and not name.startswith("--")
            and name not in KNOWN_FLAGS
            and "-" + name in KNOWN_FLAGS
        ):
            token = "-" + token
        normalized.append(token)
    return normalized


def parse_cli():
    """
    The options actually typed on the command line, as a {dest: value} dict --
    no defaults filled in, so that whatever the file's directives say can still
    show through where the command line said nothing.
    """
    opts = vars(CLI_PARSER.parse_args(normalize_flags(sys.argv[1:])))
    conflict = check_conflicts(opts)
    if conflict:
        CLI_PARSER.error(conflict)
    return opts


def check_conflicts(opts):
    """
    The option combinations that cannot be honoured, wherever they came from:
    a returned message means "these two cannot both be set". Kept apart from
    argparse's own mutually-exclusive-group machinery because the two options in
    question need not come from the same layer -- -b from a GROUND-ALL and -t
    from the command line conflict just as surely as two typed flags do.
    """
    flag_mode_opts = [
        d
        for d in (
            "ref_compile_args",
            "test_compile_args",
            "additional_test_compile_args",
        )
        if opts.get(d) is not None
    ]
    if flag_mode_opts and opts.get("baseline_model") is not None:
        return "-b/--baseline-model cannot be combined with -r/-t/-a."
    if (
        opts.get("test_compile_args") is not None
        and opts.get("additional_test_compile_args") is not None
    ):
        return "use either -t or -a, not both."
    return None


def split_directive(marker, model_path, lineno, text):
    try:
        return normalize_flags(shlex.split(text))
    except ValueError as e:
        sys.exit(
            f'ERROR: cannot read the "{marker}" directive at '
            f"{model_path}:{lineno}: {e}."
        )


def parse_directive(marker, model_path, lineno, tokens):
    try:
        opts = vars(DIRECTIVE_PARSER.parse_args(tokens))
    except DirectiveError as e:
        sys.exit(
            f'ERROR: bad "{marker}" directive at {model_path}:{lineno}: {e}\n'
            f"  It takes {os.path.basename(__file__)}'s own options, spelled "
            f"exactly as on the command line."
        )
    for dest in COMMAND_LINE_ONLY:
        if dest in opts:
            sys.exit(
                f'ERROR: the "{marker}" directive at {model_path}:{lineno} '
                f"{command_line_only_complaint(dest)}, which only the command "
                f"line may do."
            )
    conflict = check_conflicts(opts)
    if conflict:
        sys.exit(
            f'ERROR: the "{marker}" directive at {model_path}:{lineno} is '
            f"contradictory: {conflict}"
        )
    return opts


def scan_directives(model_path):
    """
    Collect the file's own options: the file-wide GROUND-ALL, and one option set
    per function that a GROUND-THIS precedes. Both markers may appear on several
    lines, whose options are simply gathered in order -- a file needing a long
    option list should be able to wrap it rather than run off the page. The
    per-function sets come back as {func: options}.
    """
    if not os.path.exists(model_path):
        sys.exit(f'ERROR: file "{model_path}" does not exist.')

    all_tokens = []
    all_lineno = None
    tokens_by_func = {}
    pending_tokens = []
    pending_lineno = None
    seen_func = False

    with open(model_path) as f:
        for lineno, line in enumerate(f, 1):
            m = GROUND_ALL_RE.match(line)
            if m:
                if seen_func:
                    sys.exit(
                        f'ERROR: the "{GROUND_ALL}" directive at '
                        f"{model_path}:{lineno} comes after a function. It sets "
                        f"the whole file's options, so it belongs in the header, "
                        f'before the first function -- use "{GROUND_ONE}" for '
                        f"options meant for one function only."
                    )
                all_tokens += split_directive(
                    GROUND_ALL, model_path, lineno, m.group(1)
                )
                if all_lineno is None:
                    all_lineno = lineno
                continue
            m = GROUND_ONE_RE.match(line)
            if m:
                pending_tokens += split_directive(
                    GROUND_ONE, model_path, lineno, m.group(1)
                )
                if pending_lineno is None:
                    pending_lineno = lineno
                continue
            m = FUNC_RE.match(line)
            if m:
                seen_func = True
                if pending_tokens:
                    tokens_by_func.setdefault(
                        m.group(1), (pending_lineno, pending_tokens)
                    )
                    pending_tokens = []
                    pending_lineno = None

    if pending_tokens:
        sys.exit(
            f'ERROR: the "{GROUND_ONE}" directive at '
            f"{model_path}:{pending_lineno} is not followed by any function, so "
            f"there is nothing for it to apply to."
        )

    ground_all = (
        parse_directive(GROUND_ALL, model_path, all_lineno, all_tokens)
        if all_tokens
        else {}
    )
    per_func = {
        func: parse_directive(GROUND_ONE, model_path, lineno, tokens)
        for func, (lineno, tokens) in tokens_by_func.items()
    }
    return ground_all, per_func


def merge_options(*layers):
    """
    Lay option sets over each other, least specific first, and remember which
    layer each surviving option came from -- so that what is in effect can be
    reported with its provenance, and a contradiction can name where each of
    its halves was set.
    """
    merged = {}
    origin = {}
    for label, opts in layers:
        for dest, value in opts.items():
            merged[dest] = value
            origin[dest] = label
    return merged, origin


def render_options(opts, origin=None):
    """
    An option set as it would be typed. FLAG_BY_DEST's order is the order the
    options are declared in, so two of these listings can be compared by eye.
    """
    parts = []
    for dest, (flag, takes_value) in FLAG_BY_DEST.items():
        if dest not in opts:
            continue
        if not takes_value:
            part = flag
        else:
            value = str(opts[dest])
            # Same "--flag=value" reason as everywhere else: a value starting
            # with "-" would be read back as a flag of its own.
            part = f"{flag}={value}" if value.startswith("-") else f"{flag} {value}"
        if origin:
            part += f" ({origin[dest]})"
        parts.append(part)
    return " ".join(parts)


def option_origins(dests, merged, origin):
    return ", ".join(
        f"{FLAG_BY_DEST[d][0]} from the {origin[d]}" for d in dests if d in merged
    )


def effective_args(cli_opts, ground_all, own_opts):
    """
    The options in effect for one function: the built-in defaults, then what the
    file says for every function, then what it says for this one, then what was
    typed. Returns the resulting namespace plus the specified-somewhere options
    and their provenance, for reporting.
    """
    merged, origin = merge_options(
        (GROUND_ALL, ground_all),
        (GROUND_ONE, own_opts),
        (COMMAND_LINE, cli_opts),
    )
    opts = dict(DEFAULTS)
    opts.update(merged)
    return argparse.Namespace(**opts), merged, origin


def default_baseline_path(model_path):
    base, ext = os.path.splitext(model_path)
    return base + BASELINE_TAG + ext


def list_functions(model_path):
    if not os.path.exists(model_path):
        sys.exit(f'ERROR: file "{model_path}" does not exist.')
    names = []
    with open(model_path) as f:
        for line in f:
            m = FUNC_RE.match(line)
            if m and m.group(1) not in names:
                names.append(m.group(1))
    if not names:
        sys.exit(f'ERROR: no function found in "{model_path}", nothing to test.')
    return names


def format_cmd(cmd, redirect_to=None):
    line = "+ " + " ".join(shlex.quote(c) for c in cmd)
    if redirect_to:
        line += " > " + shlex.quote(redirect_to)
    return line


def run_cmd(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def isolate_function(logger, src_file, func_name, dest_file):
    if not os.path.exists(src_file):
        raise SetupError(f'ERROR: file "{src_file}" does not exist.')
    cmd = [sys.executable, FIX_LIT_TEST, "-m", func_name, src_file]
    logger.log(format_cmd(cmd, redirect_to=dest_file))
    result = run_cmd(cmd)
    if result.returncode != 0 or "module {" not in result.stdout:
        logger.log("--- fixLitTest.py stdout ---")
        logger.log(result.stdout)
        logger.log("--- fixLitTest.py stderr ---")
        logger.log(result.stderr)
        logger.dump_on_failure()
        raise SetupError(
            f'ERROR: could not isolate function "{func_name}" from "{src_file}" '
            f"via fixLitTest.py -m. See log above."
        )
    logger.log(result.stderr, end="")
    with open(dest_file, "w") as f:
        f.write(result.stdout)


def strip_comments(text):
    """
    Drop every "//" comment, and every line left blank once its comment is
    gone, so that what remains is only MLIR that actually gets compiled. Both
    the "do the two variants really differ" check and the -d listing work on
    this form: a comment can then neither make two identical modules look
    different nor add noise to the diff. A "//" inside a string literal (a
    file name in some attribute, say) is left alone.
    """
    out = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut = len(line)
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
            elif ch == "\\" and in_string:
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and line.startswith("//", i):
                cut = i
                break
        stripped = line[:cut].rstrip()
        if stripped:
            out.append(stripped)
    return "".join(line + "\n" for line in out)


def read_stripped(path):
    with open(path) as f:
        return strip_comments(f.read())


def show_diff(text_a, label_a, text_b, label_b):
    print(f"--- diff: {label_a}  |  {label_b} ---")
    icdiff = shutil.which("icdiff")
    # The comment-free text is diffed out of a scratch directory: it is derived
    # data, so it has no business sitting among the kept files, where it would
    # look like something to edit and rerun.
    tmp_dir = tempfile.mkdtemp(prefix="glt_diff_")
    try:
        file_a = os.path.join(tmp_dir, "baseline.mlir")
        file_b = os.path.join(tmp_dir, "test.mlir")
        for path, text in ((file_a, text_a), (file_b, text_b)):
            with open(path, "w") as f:
                f.write(text)
        if icdiff:
            cmd = [icdiff, "--label", label_a, "--label", label_b, file_a, file_b]
        else:
            cmd = ["diff", "-y", "--label", label_a, "--label", label_b, file_a, file_b]
        # The diff writes straight to our stdout, which is block-buffered as
        # soon as it is a pipe rather than a terminal -- so flush first, or the
        # diff lands ahead of everything printed before it.
        sys.stdout.flush()
        # A nonzero return code here just means "the files differ", which is
        # expected and not itself a failure of this tool.
        subprocess.run(cmd, check=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print()


def run_onnx_model(logger, extra_args):
    cmd = [sys.executable, RUN_ONNX_MODEL] + extra_args
    logger.log(format_cmd(cmd))
    result = run_cmd(cmd)
    logger.log(result.stdout, end="")
    if result.stderr:
        logger.log(result.stderr, end="")
    return result.returncode == 0


def forwarded_input_args(args):
    extra = []
    if args.seed is not None:
        extra += ["--seed", args.seed]
    if args.shape_info is not None:
        extra += ["--shape-info", args.shape_info]
    if args.lower_bound is not None:
        extra += ["--lower-bound", args.lower_bound]
    if args.upper_bound is not None:
        extra += ["--upper-bound", args.upper_bound]
    if args.input_value is not None:
        extra += ["--input-value", args.input_value]
    return extra


def clean_kept_files():
    # Remove any files kept from a previous run -- or from the previous
    # function of this run -- before doing anything else. Otherwise, a
    # comparison that fails partway through could leave a misleading mix of
    # stale files alongside whatever it managed to produce: e.g. a fresh
    # glt_test.mlir sitting next to a stale glt_ref/ from a different
    # comparison entirely.
    for path in (GLT_TEST_FILE, GLT_BASELINE_FILE):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(GLT_REF_DIR, ignore_errors=True)


def join_args(*parts):
    return " ".join(p for p in parts if p).strip()


def resolve_mode(args, merged, origin):
    """
    File mode or flag mode, for one function's resolved options. The options
    behind that choice can now come from three places, so a contradiction says
    which place each half of it came from -- "-b from the GROUND-ALL" is a much
    shorter path to the fix than the flag alone.
    """
    conflict = check_conflicts(vars(args))
    if conflict:
        where = option_origins(
            (
                "baseline_model",
                "ref_compile_args",
                "test_compile_args",
                "additional_test_compile_args",
            ),
            merged,
            origin,
        )
        raise SetupError(f"ERROR: {conflict} In effect here: {where}.")
    return any(
        getattr(args, dest) is not None
        for dest in (
            "ref_compile_args",
            "test_compile_args",
            "additional_test_compile_args",
        )
    )


def resolve_compile_args(args, flag_mode):
    """
    The onnx-mlir options each of the two variants is compiled with.
    """
    common_compile_args = args.compile_args or ""
    if not flag_mode:
        return common_compile_args, common_compile_args

    ref_compile_args = join_args(common_compile_args, args.ref_compile_args)
    if args.test_compile_args is not None:
        test_compile_args = join_args(common_compile_args, args.test_compile_args)
    elif args.additional_test_compile_args is not None:
        test_compile_args = join_args(
            ref_compile_args, args.additional_test_compile_args
        )
    else:
        # -t defaults to empty, same as -r does when -r is absent, so
        # test = c (not ref = c + r) when neither -t nor -a is given.
        test_compile_args = common_compile_args
    if set(ref_compile_args.split()) == set(test_compile_args.split()):
        raise SetupError(
            "ERROR: in flag mode, reference and test resolve to the same "
            f"onnx-mlir options ({ref_compile_args!r}) -- there is nothing "
            "to compare. Set -t/-a to genuinely different options, or use "
            "file mode (-b) instead if the difference is in the MLIR "
            "source rather than the compiler flags."
        )
    return ref_compile_args, test_compile_args


def compare_function(
    args, flag_mode, func, ref_compile_args, test_compile_args, own_opts
):
    """
    Compare the two variants of one function; return True if they match. Raises
    SetupError if the comparison could not be carried out at all.
    """
    clean_kept_files()

    if flag_mode:
        baseline_model = args.model
        print(
            f"Comparing function {func!r} from file {args.model!r}, "
            f"compiled with onnx-mlir options {ref_compile_args!r} (baseline) "
            f"vs {test_compile_args!r} (test)."
        )
    else:
        baseline_model = args.baseline_model or default_baseline_path(args.model)
        summary = (
            f"Comparing function {func!r} from file {baseline_model!r} "
            f"(baseline) with the same function from file {args.model!r} (test)"
        )
        if ref_compile_args:
            summary += f", both compiled with onnx-mlir options {ref_compile_args!r}"
        print(summary + ".")
    # Only what this one function's own directive adds: the file-wide and typed
    # options are reported once, for the whole run.
    if own_opts:
        print(f"{GROUND_ONE} options: {render_options(own_opts)}")

    if not flag_mode and not os.path.exists(baseline_model):
        raise SetupError(
            f"ERROR: no baseline available.\n"
            f'  Tried baseline file "{baseline_model}" (not found).\n'
            f"  Either create that file (same function name as --func), "
            f"or pass -b/--baseline-model, or pass -r/-t/-a "
            f"to compare the test file against itself with different options."
        )

    logger = Logger(args.verbose)
    try:
        isolate_function(logger, args.model, func, GLT_TEST_FILE)

        if flag_mode:
            baseline_isolated = GLT_TEST_FILE
        else:
            baseline_isolated = GLT_BASELINE_FILE
            isolate_function(logger, baseline_model, func, GLT_BASELINE_FILE)

        if args.diff:
            if flag_mode:
                print("--- diff: reference compile args  |  test compile args ---")
                print(f"  reference: {ref_compile_args!r}")
                print(f"  test:      {test_compile_args!r}")
                print()
            else:
                show_diff(
                    read_stripped(baseline_isolated),
                    f"baseline ({baseline_model})",
                    read_stripped(GLT_TEST_FILE),
                    f"test ({args.model})",
                )

        if not flag_mode and read_stripped(baseline_isolated) == read_stripped(
            GLT_TEST_FILE
        ):
            # There is nothing to compare: both files hold the same module once
            # the comments are out of the way, so running them would only prove
            # that identical code computes identical results.
            print(
                f"FAIL: {func} is identical (ignoring comments) in "
                f"{baseline_model} (baseline) and {args.model} (test) -- the "
                f"two variants must genuinely differ for the comparison to "
                f"mean anything."
            )
            return False

        logger.log("Compiling and running the baseline/reference variant ...")
        baseline_cmd = [
            "-m",
            baseline_isolated,
            "--save-ref",
            GLT_REF_DIR,
        ]
        if ref_compile_args:
            # Use "--flag=value" (not two separate argv items) so a value that
            # itself starts with "-" (e.g. "-O0") isn't misparsed by
            # RunONNXModel.py's own argparse as a new option.
            baseline_cmd += [f"--compile-args={ref_compile_args}"]
        baseline_cmd += forwarded_input_args(args)
        if not run_onnx_model(logger, baseline_cmd):
            logger.dump_on_failure()
            raise SetupError(
                "ERROR: the baseline/reference variant failed to compile or "
                "run -- this is a setup problem, not a mismatch. See log above."
            )

        logger.log(
            "Compiling and running the test variant, verifying against the "
            "saved reference ..."
        )
        test_cmd = [
            "-m",
            GLT_TEST_FILE,
            "--load-ref",
            GLT_REF_DIR,
            "--verify",
            "ref",
            "--verify-every-value",
            "--rtol",
            args.rtol,
            "--atol",
            args.atol,
        ]
        if test_compile_args:
            test_cmd += [f"--compile-args={test_compile_args}"]
        passed = run_onnx_model(logger, test_cmd)

        print()
        # Both the recipe and the files it works on are only worth offering for
        # the one function -f asked about. Testing every function, the commands
        # would be repeated per function and all but the last one's would refer
        # to files the next function has already overwritten, so nothing is left
        # behind at all (see the end of main) -- rerun with "-f <func>" to get
        # the recipe, and the files, for whichever function turns out to be the
        # interesting one.
        if args.func is not None:
            print("To reproduce this test manually:")
            print(
                format_cmd(
                    [FIX_LIT_TEST_NAME, "-m", func, args.model],
                    redirect_to=GLT_TEST_FILE,
                )
            )
            if not flag_mode:
                print(
                    format_cmd(
                        [FIX_LIT_TEST_NAME, "-m", func, baseline_model],
                        redirect_to=GLT_BASELINE_FILE,
                    )
                )
            print(format_cmd([RUN_ONNX_MODEL_NAME] + baseline_cmd))
            print(format_cmd([RUN_ONNX_MODEL_NAME] + test_cmd))

            kept_baseline = GLT_BASELINE_FILE if not flag_mode else "<none, same file>"
            print(f"Kept files: {GLT_TEST_FILE}, {kept_baseline}, {GLT_REF_DIR}/")
            print()

        if not passed:
            logger.dump_on_failure()
            if not args.diff:
                print(
                    "Hint: rerun with -d/--diff to see exactly what differs "
                    "between the two variants -- that's the most likely "
                    "place to look for the cause."
                )

        if passed:
            print(
                f"PASS: {func} matches its baseline within "
                f"rtol={args.rtol}, atol={args.atol}."
            )
        else:
            print(
                f"FAIL: {func} does NOT match its baseline within "
                f"rtol={args.rtol}, atol={args.atol}."
            )
        return passed
    finally:
        logger.close()


def main():
    cli_opts = parse_cli()
    model = cli_opts["model"]
    ground_all, per_func_directive = scan_directives(model)

    single_func = "func" in cli_opts
    funcs = [cli_opts["func"]] if single_func else list_functions(model)
    if not single_func:
        print(
            f"Testing all {len(funcs)} function(s) of {model!r}, one at a "
            f"time: {', '.join(funcs)}."
        )
        print()

    succeeded = []
    failed = []
    for i, func in enumerate(funcs):
        if not single_func:
            print(f"=== [{i + 1}/{len(funcs)}] {func} ===")
        own_opts = per_func_directive.get(func, {})
        args, merged, origin = effective_args(cli_opts, ground_all, own_opts)
        try:
            flag_mode = resolve_mode(args, merged, origin)
            ref_compile_args, test_compile_args = resolve_compile_args(args, flag_mode)
            passed = compare_function(
                args,
                flag_mode,
                func,
                ref_compile_args,
                test_compile_args,
                own_opts,
            )
        except SetupError as e:
            # With -f, a setup problem is fatal, exactly as it was before.
            # Testing every function, it sinks only the function it happened
            # to: the remaining ones are still worth running, all the more so
            # now that a function can be misconfigured on its own by its
            # GROUND-THIS directive.
            if single_func:
                sys.exit(str(e))
            print(str(e))
            print(f"FAIL: {func} could not be compared (see above).")
            passed = False
        (succeeded if passed else failed).append(func)
        if not single_func:
            print()

    if not single_func:
        # Whatever the last function left behind belongs to that function alone,
        # and saying so would only invite reading it as this run's result.
        clean_kept_files()
        # What was analyzed, before what came of it: the full path (this run may
        # well have been started from somewhere else entirely) and the options
        # that applied to the whole file -- the typed ones and the file's own
        # GROUND-ALL, each marked with where it came from. Per-function options
        # are reported with their function, above.
        file_options, file_origin = merge_options(
            (GROUND_ALL, ground_all), (COMMAND_LINE, cli_opts)
        )
        print("=== summary ===")
        print(f"File:    {os.path.abspath(model)}")
        print(f"Options: {render_options(file_options, file_origin) or '<none>'}")
        # Both lists are in file order, so this summary can be read side by
        # side with the .mlir file it came from.
        print(f"Succeeded ({len(succeeded)}): {', '.join(succeeded) or '<none>'}")
        print(f"Failed    ({len(failed)}): {', '.join(failed) or '<none>'}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
