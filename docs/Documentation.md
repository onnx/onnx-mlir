<!--- SPDX-License-Identifier: Apache-2.0 -->

# About Documentation

## How to add a new documentation page

Firstly, `/docs` is the root directory of the documentation website, meaning that any
documentation page you wish to display to a user must be located within `/docs`.

Secondly, add the documentation page into the navigation configuration file located at
`/docs/_data/navigation.yaml`. Edit the table of content to include the path to
the newly created documentation page with a descriptive title.

Then, capture the changes done in a patch and submit a pull request; once the patch is
merged into `onnx-mlir` codebase, a link pointing to the file path specified with the
descriptive title you provided will appear on the navigation panel.