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

      # pennylane-lightning = ps.buildPythonPackage rec {
      #   pname = "pennylane-lightning";
      #   version = "0.43.0";
      #   pyproject = true;
      #
      #   src = pkgs.fetchFromGitHub {
      #     owner = "PennyLaneAI";
      #     repo = "pennylane-lightning";
      #     inherit version;
      #     tag = "v${version}";
      #     hash = "sha256-YAaUIk1dA+ZIADsLk7MT8eAogKp0O/gibD3ImzguJ60=";
      #   };
      #
      #   nativeBuildInputs = [
      #     ps.setuptools
      #     pkgs.cmake
      #     pkgs.ninja
      #     ps.tomli
      #   ];
      #
      #   propagatedBuildInputs = with ps; [
      #   ];
      # };

      pennylane-lightning = ps.buildPythonPackage rec {
        pname = "pennylane-lightning";
        version = "0.43.0";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/46/0f/7161bdc28fcbfab1341d66bbc106fc30db3d21d1caa6747994e9314655b1/pennylane_lightning-0.43.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
          hash = "sha256-a5viL4KQ4nWLePqle4x4m7RyzOVjeolqEMnIh3KqGDw=";
        };

        doCheck = false;
      };

      pennylane = ps.buildPythonPackage rec {
        pname = "pennylane";
        version = "0.43.1"; # pick the version you want
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "PennyLaneAI";
          repo = "pennylane";
          tag = "v${version}";
          sha256 = "sha256-Ab2pElCPsW2c648lAzzcZ7PQ0lffEfVetyh1757usGg=";
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
