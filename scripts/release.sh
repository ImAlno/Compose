#!/usr/bin/env bash
# Release composeai to PyPI: scripts/release.sh X.Y.Z
#
# Reads the PyPI token from .env (git-ignored):
#   TWINE_USERNAME=__token__
#   TWINE_PASSWORD=pypi-...
#
# Refuses to run on a dirty tree so the published artifacts always match a
# commit. After a successful upload, commit the version bump and tag it.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "error: '$VERSION' is not X.Y.Z" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "error: working tree is dirty — commit first" >&2; exit 1; }

if [[ -f .env ]]; then set -a; source .env; set +a; fi
[[ "${TWINE_PASSWORD:-}" == pypi-* ]] || {
    echo "error: no PyPI token — put TWINE_USERNAME=__token__ and TWINE_PASSWORD=pypi-... in .env" >&2
    exit 1
}

CURRENT=$(.venv/bin/python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
if [[ "$VERSION" == "$CURRENT" ]]; then
    # Tree was committed with the bump already in place (e.g. version bumped
    # as part of the release branch) -- nothing to rewrite, go straight to gates.
    echo "version is already $VERSION -- skipping bump"
else
    echo "bumping $CURRENT -> $VERSION"
    sed -i '' "s/^version = \"$CURRENT\"/version = \"$VERSION\"/" pyproject.toml
    sed -i '' "s/__version__ = \"$CURRENT\"/__version__ = \"$VERSION\"/" src/composeai/__init__.py
    sed -i '' "s/\"$CURRENT\"/\"$VERSION\"/" tests/test_package.py
fi

echo "running gates..."
.venv/bin/pytest -q
.venv/bin/ruff check src tests examples
.venv/bin/pyright

echo "building..."
rm -rf dist
.venv/bin/python -m build
.venv/bin/twine check dist/*

echo "uploading to PyPI..."
.venv/bin/twine upload --non-interactive dist/*

echo
echo "released composeai $VERSION — now lock it in:"
echo "    git commit -am \"Release $VERSION\" && git tag \"v$VERSION\""
