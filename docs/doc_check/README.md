<!--- SPDX-License-Identifier: Apache-2.0 -->

# DocCheck

### Goal

It is always desirable to ensure that every piece of knowledge has a single, unambiguous, authoritative representation 
in our codebase. However, sometimes violating such principle can result in improved overall quality of the software
project. For instance, when we write documentation containing example code snippets, it is desirable to write tests
for them - however, if we do so, the same code example will exist both in documentation and in tests! Such duplication
of knowledge has tangible adverse consequences - when documentation is updated to new examples, tests become obsolete. 
Moreover, the discrepancy between multiple copies of the same knowledge (e.g., code example) can only be spotted with 
manual inspection.

Under such circumstances, to establish a single source of trough in an enforceable manner, we can turn to the DocCheck
tool. Simply put, DocCheck enforces the consistency constraints as specified by the users between textual artifacts in 
our codebase. Textual artifacts can be:
- Sections in documentation
- Content of a file
- Output of command execution
- ...

Specifically, DocCheck allows us to precisely specify how a textual artifact is derived from another. Such
specification is then parsed and verified by our software testing infrastructure to ensure the consistency between
derived textual artifact and the original one. This overall workflow provides an enforceable way to establish a single,
unambiguous and authoritative representation of knowledge in our codebase.

### Directives

Directives can be used to communicate the relationship between derived and original textual artifacts to DocCheck. 
DocCheck will perform consistency constraints checking according to the specification. In this section, supported 
directives are explained in details.

Currently, directives can be specified either in a Markdown file or in a standalone DocCheck configuration file (a file 
ending with `.dc` suffix). For markdown file, specify directive using the following syntax:

```markdown
[{directive}]: <> ({configuration})
```

For standalone DocCheck configuration file, use the following syntax:
```
{directive}({configuration})
```

where `{directive}` is the name of the directive and `{configuration}` expresses the specific 
parameters of this directive. In general, a directive configuration is expressed using a python dictionary literal, 
with supported configuration parameter name as keys and the desired state of configuration as values.

Special shorthands exist for each directive individually.

##### `same-as-file`:

Use `same-as-file` directive to ensure that the code section following this directive is the same as a source file.
This is useful primarily because testing code snippet in documentation directly is often impossible. However,
unit tests can be written utilizing an exact copy of the code snippet content. We can use `same-as-file` directive 
to ensure the code snippet is always the same as its copy used in some unit tests. 

`same-as-file` directive supports a convenient short-hand configuration format where the directive configuration can 
be fully specified using the name of the reference file to check against. For example, to ensure a code snippet is the 
same as a unit-tested file `reference.cpp`, use the following directive as shown in the documentation snippet:

[same-as-file]: <> ({"ref": "docs/doc_check/test/same-as-file/simple/README.md", "skip-ref": 2})
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````

In the canonical form of directive configuration (as a python dictionary literal), this directive supports these parameters in it:

`ref` (string): reference file to check against.

`skip-doc` (int): number of lines to skip when checking the documentation.

`skip-ref` (int): number of lines to skip when scanning the reference file.

For example, to ensure the following code snippet is the same as a unit-tested file `reference.cpp`, except for the first 2 lines of the code used in documentation, and the first 3 lines of code used in the reference file, the following directive configuration can be used:

[same-as-file]: <> ({"ref": "docs/doc_check/test/same-as-file/skip-doc-ref/README.md", "skip-ref": 2})
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> ({"ref": "reference.cpp", "skip-doc": 2, "skip-ref": 3})
```cpp
// First line unique to documentation
// Second line unique to documentation
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````

#### `file-same-as-stdout`

Use `file-same-as-stdout` to ensure that file content is the same as the output of executing a command.
This directive supports these parameters in it:

`file` (string): file to compare with.

`cmd` (List[str]): the command (expressed as a list of command components), e.g. `["ls", "-l"]`.

For example, to ensure that the content of a file `test.in`:

[same-as-file]: <> (docs/doc_check/test/file-same-as-stdout/success/test.in)
```
dog
```

is exactly the same as the output of command execution `echo dog`, one can use the following directive:
[same-as-file]: <> (docs/doc_check/test/file-same-as-stdout/success/test.in.dc)
```
file-same-as-stdout({"file": "test.in", "cmd": ["echo", "dog"]})
```