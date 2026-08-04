{
  description = "A flake for running GRAHF and the hm-ibf-hao horizontal alignment benchmark.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/18dd725c29603f582cf1900e0d25f9f1063dbf11";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (
    system:
    let
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      # Rust toolchain including the components the verification gate needs.
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
      };

      # Python environment for the preprocessing scripts and their tests.
      pythonEnv = pkgs.python311.withPackages (ps: with ps; [
        numpy
        scipy
        pytest
      ]);

      # Fixed version of `irace` from GitHub.
      irace_dev = pkgs.rPackages.buildRPackage {
        name = "irace";
        src = builtins.fetchGit {
          url = "https://github.com/MLopez-Ibanez/irace";
          rev = "b790572ecc4e79c2fe83417dbd51a30ddfad6e60";
        };
        propagatedBuildInputs = with pkgs.rPackages; [ matrixStats R6 withr ];
      };
      # Fixed version of `iracepy-tiny` from GitHub.
      iracepy_tiny = pkgs.python311Packages.buildPythonPackage {
        pname = "iracepy-tiny";
        format = "pyproject";
        version = "0.1.0";
        src = builtins.fetchGit {
          url = "https://github.com/Saethox/iracepy-tiny";
          rev = "8e22a17fba9040a4b56b9f07cbe6082be0a2fd7b";
        };
        nativeBuildInputs = with pkgs.python311Packages; [ hatchling ];
        propagatedBuildInputs = with pkgs.python311Packages; [ numpy scipy pandas rpy2 ];
      };
      # Fixed version of `enoppy` from GitHub.
      enoppy = pkgs.python311Packages.buildPythonPackage {
        pname = "enoppy";
        pyproject = true;
        version = "0.1.1";
        src = builtins.fetchGit {
          url = "https://github.com/Saethox/enoppy";
          ref = "fix-divide-zero";
          rev = "3cedf89a5071ec7cd002dbfb1f7cdd165130b95b";
        };
        nativeBuildInputs = with pkgs.python311Packages; [ setuptools ];
        propagatedBuildInputs = with pkgs.python311Packages; [ numpy scipy ];
        dontCheckRuntimeDeps = true;
      };
    in {
      devShells.default = pkgs.mkShell {
        name = "hm-ibf-hao-shell";
        nativeBuildInputs = [ pkgs.bashInteractive ];
        buildInputs = with pkgs; [
          # irace R package, required by `irace-rs` during training.
          R irace_dev
          # iracepy-tiny Python wrapper + Enoppy.
          iracepy_tiny enoppy
          # Python runtime and test tooling for the pipeline scripts.
          pythonEnv ruff
          # C bindgen.
          llvmPackages.libclang.lib futhark clang
          # Rust toolchain with clippy and rustfmt.
          rustToolchain
        ];

        # C bindgen.
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        # Print entire error backtrace.
        RUST_BACKTRACE = "full";
        RUST_LOG = "info";
      };
    });
}
