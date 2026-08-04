#!/usr/bin/env bash
#
# The verification gate defined in AGENTS.md ("Definition of done").
#
# Runs inside the container built from `Dockerfile`; use `./run.sh verify` from the host.
# Every step covers the entire workspace, not just the files that were touched.
#
set -euo pipefail

cd "$(dirname "$0")/.."

step() {
    echo
    echo "==> $*"
}

step "cargo fmt --check"
cargo fmt --all -- --check

step "cargo clippy -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

step "cargo test"
cargo test --workspace

step "ruff check"
ruff check .

step "ruff format --check"
ruff format --check .

step "pytest"
python3 -m pytest

echo
echo "==> All checks passed."
