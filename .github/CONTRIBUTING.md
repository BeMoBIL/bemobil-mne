# Contributing to `data-analysis`

Thanks for contributing! This guide describes how we develop and how we cut a
release.

## How to develop

- All changes should be pushed to this repo only via pull requests (PRs).
- PRs should be merged only via the **"Squash and merge"** option.
  - Like this, each commit to `main` refers to a PR.
  - GitHub allows us to link back to that PR, so all conversation, review, and
    individual commits stay visible from there.
  - This keeps a clean (tidy) and transparent commit history on `main`.
  - PRs should have meaningful titles, as they will end up as the squashed
    commit message on `main`.
  - PRs should be atomic: two unrelated changes should not be done in the same
    PR, if possible and sensible.
- Within each PR, the submitter is responsible for keeping a clean branch and
  commit history.
  - Commits should be atomic and have meaningful descriptions to make review
    easier.

## How to make a release

1. Make sure you have a git remote `upstream` configured to point to
   <https://github.com/BeMoBIL/data-analysis>.
2. From your (clean!) `main` branch, run:
   1. `git fetch --all --tags --prune --prune-tags`
   2. `git rebase upstream/main`
   3. `git push`
3. Tag the current version, following [semantic versioning](https://semver.org/).
   For example, for `v0.1.0`, use:
   `git tag -a -m "v0.1.0" v0.1.0 upstream/main`
   (always prepend a `v` to the version!).
4. Push the tag upstream: `git push --follow-tags upstream`.
5. Make a release on GitHub: select the pushed tag, name the release after the
   tag, and let GitHub generate the release notes automatically. Add the note
   from step 5 at the top of the release notes.
