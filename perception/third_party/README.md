# Third-party Code

This directory is for vendored external code that is used by perception tasks but is not maintained as first-party project code.

## Current Contents

- `sort/`: vendored SORT multi-object tracker implementation, including its upstream README, license, requirements, and sample MOT data.

## Guidelines

- Keep each dependency in its own subdirectory.
- Preserve upstream license files and README files.
- Avoid editing vendored code unless the local change is documented.
- Prefer adding a short note here when a new third-party dependency is added, including why it is vendored instead of installed from package metadata.
- Keep large generated outputs and local datasets out of this directory.

