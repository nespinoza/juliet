## What this changes

## Category

Pick one. A PR that needs more than one should be split.

- [ ] **A: pure refactor.** No numerical change. No baseline or xfail touched.
- [ ] **B: crash to working.** Something that raised now works. The matching `xfail` is removed. No previously valid output changes.
- [ ] **C: science change.** Valid inputs give different numbers. Approved by the maintainer, with a "Changed results" entry in `CHANGELOG.md` and a regenerated baseline whose reason is in `tests/golden/CHANGES.md`.
- [ ] **New capability.** New configurations added to `tests/golden/configs.py`, with no existing baseline changed.

## Checks

- [ ] `pytest` passes locally.
