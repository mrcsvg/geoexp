# Releasing geoexp

Runbook for publishing a release to PyPI. The publish itself is automated by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) via PyPI
Trusted Publishing; this document covers the one-time setup and the
per-release steps. It is the augsynth-py runbook with names swapped.

## One-time setup (before the first release)

1. Re-verify the `geoexp` name is still free on PyPI (it was on 2026-08-12).
2. Register the **pending publisher** — because the project does not exist
   on PyPI yet, this is done under *Account settings → Publishing → Add a
   new pending publisher* with:
   - PyPI project name: `geoexp`
   - Owner: `mrcsvg`
   - Repository: `geoexp`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repo, create the `pypi` environment
   (*Settings → Environments → New environment*). Adding yourself as a
   required reviewer is recommended — it turns every publish into a
   one-click manual approval.

No API tokens are created or stored anywhere; the workflow authenticates via
OIDC (trusted publishing).

## Per-release steps

1. Make sure `main` is green (CI **and** the validation-against-R workflow).
2. Update `src/geoexp/_version.py` to the release version (the release
   workflow refuses to publish when the tag and `_version.py` disagree).
3. Move the `[Unreleased]` items in `CHANGELOG.md` under a new
   `[X.Y.Z] - YYYY-MM-DD` heading and update the link references at the
   bottom.
4. Land those changes on `main` via the normal PR flow.
5. Tag and push:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "geoexp X.Y.Z"
   git push origin vX.Y.Z
   ```

6. The `Release to PyPI` workflow runs: lint + unit-test gate, build,
   `twine check`, then publish (pausing for approval if the `pypi`
   environment has required reviewers).
7. Verify: `pip install geoexp==X.Y.Z` in a clean venv and run the README
   example.
8. Create a GitHub Release for the tag (paste the changelog section).

## Versioning policy

Semantic versioning with the 0.x caveat: minor bumps may break the API;
patch bumps must not. See the version map in `CLAUDE.md` for how geoexp
releases track the project milestones (v0.5 → 0.1.x, v0.6 → 0.3.x+).
