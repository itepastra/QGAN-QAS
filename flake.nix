{
  description = "Python development shell for QGAN and QAS";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      py = pkgs.python3; # strongly suggest 3.12 here
      ps = py.pkgs;

      autoray = ps.buildPythonPackage rec {
        pname = "autoray";
        version = "0.8.0";
        pyproject = true;

        src = ps.fetchPypi {
          inherit pname version;
          hash = "sha256-XQ1x2gPLAtW8WQoa9k4LpYWJNS1iiEOg7Lz+kAQNxSA=";
        };

        nativeBuildInputs = [
          ps.hatchling
          ps.hatch-vcs
        ];
      };

      diastatic-malt = ps.buildPythonPackage rec {
        pname = "diastatic-malt";
        version = "2.15.2";
        pyproject = true;

        src = ps.fetchPypi {
          inherit pname version;
          hash = "sha256-frkNjDC3/xa06Ew6Zd4v9/W3udD1zeojkY50f/f7UyA=";
        };

        nativeBuildInputs = [
          ps.setuptools
        ];

        propagatedBuildInputs = [
          ps.astunparse
          ps.gast
          ps.termcolor
        ];
      };

      scipy_openblas32_compat = pkgs.runCommand "scipy-openblas32-compat" { } ''
        mkdir -p $out/lib
        # CMake is specifically looking for libscipy_openblas.so
        ln -s ${pkgs.openblas}/lib/libopenblas.so $out/lib/libscipy_openblas.so
      '';

      pennylane-lightning =
        let
          nanobind_src = pkgs.fetchFromGitHub {
            owner = "wjakob";
            repo = "nanobind";
            tag = "v2.8.0";
            hash = "sha256-GGYnyO8eILYNu7va2tMB0QJkBCRDMIfRQO4a9geV49Y=";
            fetchSubmodules = true;
          };

          catalyst_src = pkgs.fetchFromGitHub {
            owner = "PennyLaneAI";
            repo = "catalyst";
            rev = "d253c59c06728f29b6703e3c9d478e9d7d5823be";
            hash = "sha256-bDYQWY7i4PN/hckuicUgksZKl9WhDslaxYtk5o6zMMs=";
            # if catalyst itself uses submodules/fetchcontent, you may need:
            # fetchSubmodules = true;
          };
        in
        ps.buildPythonPackage rec {
          pname = "pennylane-lightning";
          version = "0.43.0";
          pyproject = true;

          src = pkgs.fetchFromGitHub {
            owner = "PennyLaneAI";
            repo = "pennylane-lightning";
            tag = "v${version}";
            hash = "sha256-YAaUIk1dA+ZIADsLk7MT8eAogKp0O/gibD3ImzguJ60=";
            fetchSubmodules = true;
          };

          # IMPORTANT: prevent Nix’s cmake setup-hook from taking over and leaving us in ./build
          dontConfigure = true;

          build-system = with ps; [
            setuptools
            wheel
            cmake
            ninja
            tomli
          ];

          nativeBuildInputs = [
            # pkgs.cmake
            # pkgs.ninja
            pkgs.pkg-config
          ];

          dontCheckRuntimeDeps = true;

          # If you need OpenBLAS compat for SciPy/OpenBLAS symbol name:
          preBuild = ''
            export SCIPY_OPENBLAS32=${scipy_openblas32_compat}/lib

            # Pass CMake args to the backend-driven CMake run (scikit-build/setuptools integrations read CMAKE_ARGS)
            export CMAKE_ARGS="$CMAKE_ARGS -DFETCHCONTENT_SOURCE_DIR_NANOBIND=${nanobind_src} -DLIGHTNING_CATALYST_SRC_PATH=${catalyst_src}"
          '';

        };

      pennylane = ps.buildPythonPackage rec {
        pname = "pennylane";
        version = "0.43.1"; # pick the version you want
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "PennyLaneAI";
          repo = "pennylane";
          tag = "v${version}";
          hash = "sha256-Ab2pElCPsW2c648lAzzcZ7PQ0lffEfVetyh1757usGg=";
        };

        nativeBuildInputs = with ps; [
          setuptools
        ];

        propagatedBuildInputs = with ps; [
          # plus whatever else the build complains about missing
          scipy
          networkx
          rustworkx
          autograd
          appdirs
          cachetools
          requests
          tomlkit
          typing-extensions
          numpy

          autoray
          diastatic-malt
          pennylane-lightning
        ];

        # doCheck = false;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (py.withPackages (
            python-pkgs: with python-pkgs; [
              numpy
              tensorflow
              pennylane
            ]
          ))
        ];
      };
    };
}
