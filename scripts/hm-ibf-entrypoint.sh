#!/usr/bin/env bash
#
# Installed on PATH as `hm-ibf` inside the dev image (see Dockerfile). Builds the release
# binary if needed and always runs it from hm-ibf-hao/, so its relative defaults
# (instances/, hao_run/, results/) resolve correctly no matter which directory you call
# it from.
#
#   hm-ibf --help
#   hm-ibf train
#   hm-ibf evaluate
#   hm-ibf pipeline
#
# `train` and `pipeline` call IRACE, which needs R and the `irace` package from the Nix
# flake. If `nix` is on PATH (the `dev-nix` image) and we are not already inside a Nix
# shell, re-exec through `nix develop` automatically so those subcommands work with no
# extra steps; otherwise they fail with a clear ModuleNotFoundError from irace-rs.
set -euo pipefail

case "${1:-}" in
    train | pipeline)
        if [ -z "${IN_NIX_SHELL:-}" ] && command -v nix >/dev/null 2>&1; then
            exec nix develop /app --command "$0" "$@"
        fi
        ;;
esac

cd /app/hm-ibf-hao
cargo build --release --bin hm-ibf-hao -p grahf-hao
exec /app/target/release/hm-ibf-hao "$@"
