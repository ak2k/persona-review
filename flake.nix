{
  description = "Run a compound-engineering review persona through grok or codex and get schema-valid findings back, not a transcript";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    # Not eachDefaultSystem: that includes x86_64-darwin, which nixpkgs 26.11 dropped, so
    # `nix flake check --all-systems` failed on evaluation alone rather than on anything in
    # this repo. Declare what is actually supported.
    flake-utils.lib.eachSystem
      [
        "aarch64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ]
      (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3;

          # The suites' own dependencies. basedpyright needs this env too, not just a bare
          # interpreter: `typeCheckingMode: strict` reports Unknown member types for an
          # unresolvable `import pytest`, so a type check run without it would fail on every
          # `pytest.raises` in tests/ — or, with reportMissingImports off, pass while
          # inferring Unknown for all of them.
          pythonEnv = python.withPackages (ps: [
            ps.pytest
            ps.pytest-cov
            ps.hypothesis
          ]);

          # `grok` and `codex` are deliberately NOT dependencies. They are the user's own
          # subscription-authenticated CLIs, resolved from PATH at run time so a review uses
          # the login they already have rather than an API key this package would need.
          persona-review = python.pkgs.buildPythonApplication {
            pname = "persona-review";
            version = "0.3.1";
            pyproject = true;
            src = ./.;

            build-system = [ python.pkgs.setuptools ];

            # No test phase here: the suites need a writable HOME and stub runners on PATH,
            # and they are run as their own checks below against THIS built output.
            doCheck = false;

            # The gate-replacement hole is ONE mechanism: the wrapper adds this package's
            # site-packages with `site.addsitedir`, which APPENDS, so anything landing
            # earlier on sys.path shadows `persona_review` wholesale — the planted module
            # answers, prints a plausible summary, exits 0, and runs as the user outside the
            # read-only sandbox, with the model never invoked.
            #
            # The cwd, PYTHONPATH and PYTHONHOME are three INSTANCES of that mechanism, not
            # the set. Two were closed one at a time under a comment reading "TWO doors, and
            # both have to be shut"; a reviewer promptly found the third. So the doors below
            # are defence in depth, and the actual guarantee is PERSONA_REVIEW_LIB: the code
            # asserts at startup that the gate it is running came from here, which holds for
            # instances nobody has enumerated.
            #
            # Safe to unset PYTHONPATH/PYTHONHOME: this package has no dependencies and the
            # wrapper puts its own site-packages on the path explicitly.
            makeWrapperArgs = [
              "--set"
              "PERSONA_REVIEW_LIB"
              "${placeholder "out"}/${python.sitePackages}"
              "--unset"
              "PYTHONPATH"
              "--unset"
              "PYTHONHOME"
              "--set"
              "PYTHONNOUSERSITE"
              "1"
              "--set"
              "PYTHONSAFEPATH"
              "1"
            ];

            meta = {
              description = "Schema-validated code review through grok / codex persona briefs";
              license = pkgs.lib.licenses.asl20;
              platforms = pkgs.lib.platforms.unix;
              mainProgram = "ce-grok-persona";
            };
          };
        in
        {
          packages = {
            default = persona-review;
            persona-review = persona-review;
          };

          devShells.default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.ruff
              pkgs.basedpyright
              pkgs.git
            ];
            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };

          checks = {
            # Drives the INSTALLED console scripts as processes against stub runners: exit
            # status, refusals before dispatch, artifact lifecycle, timeouts, and the
            # one-line stdout contract. This is the reason the code lives in its own repo —
            # a check that reads the source proves the files are fine, while this proves
            # what actually ships is.
            process =
              pkgs.runCommand "persona-review-process"
                {
                  nativeBuildInputs = [
                    pythonEnv
                    pkgs.coreutils
                    # The size preflight measures `git diff BASE..HEAD`, so proving it fires
                    # needs a real repository with a real large diff.
                    pkgs.git
                  ];
                }
                ''
                  export HOME=$(mktemp -d)
                  # -p no:cacheprovider: rootdir is a read-only store path and pytest writes
                  # .pytest_cache there by default. --no-cov because every case here runs the
                  # library in a SUBPROCESS, so measuring the parent would report ~0% and
                  # trip the gate for a suite that is doing its job.
                  PERSONA_REVIEW_BIN=${persona-review}/bin \
                    python3 -m pytest ${self}/tests/test_process.py \
                      -p no:cacheprovider --no-cov --no-header -q
                  touch $out
                '';

            unit =
              pkgs.runCommand "persona-review-unit"
                {
                  nativeBuildInputs = [ pythonEnv ];
                  PYTHONDONTWRITEBYTECODE = "1";
                }
                ''
                  export HOME=$(mktemp -d)
                  # The library under test is the PACKAGED copy, so the modules that ship are the
                  # modules exercised. PERSONA_REVIEW_EXPECT_LIB makes that claim testable from
                  # inside the process, because only the process knows what it actually imported:
                  # an earlier suite put its own source root ahead of PYTHONPATH and passed with
                  # every packaged module replaced by `raise RuntimeError`.
                  # Negative control FIRST, because the guard silently no-ops one line away:
                  # dropping PERSONA_REVIEW_EXPECT_LIB while PYTHONPATH is wrong makes this check
                  # pass while the suite imports the source tree — the exact failure the comment
                  # above describes, restorable by a plausible refactor. Prove the guard bites
                  # before believing it about the real run.
                  if PYTHONPATH=/nonexistent-sitepackages \
                     PERSONA_REVIEW_EXPECT_LIB=${persona-review} \
                       python3 -m pytest ${self}/tests/test_unit.py \
                         -p no:cacheprovider --no-cov --no-header -q > control.log 2>&1; then
                    echo "FAIL: the suite passed while importing something other than the" >&2
                    echo "packaged library. PERSONA_REVIEW_EXPECT_LIB is not being honoured." >&2
                    exit 1
                  fi
                  grep -q 'not the packaged library' control.log || {
                    echo "FAIL: the control run failed, but not on the packaged-library guard:" >&2
                    cat control.log >&2
                    exit 1
                  }
                  echo "ok: the packaged-library guard rejects a suite importing the source tree"

                  # The one check where coverage is meaningful, so it is the one that carries
                  # the gate: the process and mutation suites drive the library through
                  # subprocesses, which the parent's tracer cannot see.
                  #
                  # No explicit --cov here. pyproject's addopts already says
                  # `--cov=persona_review`, which coverage resolves as a PACKAGE — the built
                  # one, via PYTHONPATH below. Passing the store path as well would measure
                  # both it and the never-imported source copy, reporting the latter at 0%
                  # and failing the gate for a suite that is fully covering what it tests.
                  #
                  # --cov-config IS needed, though: pytest finds its own ini by walking up
                  # from the test file, but coverage looks for [tool.coverage.run] in the
                  # CURRENT DIRECTORY, which here is an empty build dir. Without this the
                  # sandbox silently ran with branch coverage off and the omit list empty.
                  PYTHONPATH=$(echo ${persona-review}/${python.sitePackages}) \
                  PERSONA_REVIEW_EXPECT_LIB=${persona-review} \
                    python3 -m pytest ${self}/tests/test_unit.py \
                      -p no:cacheprovider --no-header -q \
                      --cov-config=${self}/pyproject.toml
                  touch $out
                '';

            # Every guard must be able to fail. The recurring defect in this package is not
            # a wrong guard but a guard that CANNOT fail — seven shipped green in one review
            # cycle, each found by a reviewer running mutations by hand. This reverts each
            # fix in a scratch copy and requires a test to die, so a future guard has to
            # earn its place rather than merely exist.
            mutations =
              pkgs.runCommand "persona-review-mutations"
                {
                  nativeBuildInputs = [
                    pythonEnv
                    pkgs.coreutils
                    # The PROCESS tier of the harness runs tests/test_process.py, which
                    # symlinks a real git into its stub directory to exercise the size
                    # preflight. The unit tier needs none of this.
                    pkgs.git
                  ];
                  PYTHONDONTWRITEBYTECODE = "1";
                }
                ''
                  export HOME=$(mktemp -d)
                  # --no-cov: the harness spawns a suite per mutation, so the parent covers
                  # nothing.
                  #
                  # PERSONA_REVIEW_PKG makes it mutate a scratch copy of the BUILT package
                  # rather than the source tree, so the guards it proves can fail are the
                  # ones that ship. The unit and process checks already run the built output;
                  # a harness measuring a different copy is the same defect their own
                  # negative controls exist to catch. The suite asserts this variable is
                  # honoured, so the source-tree fallback cannot silently apply here.
                  PERSONA_REVIEW_PKG=${persona-review}/${python.sitePackages} \
                    python3 -m pytest ${self}/tests/test_mutations.py \
                      -p no:cacheprovider --no-cov --no-header -q
                  touch $out
                '';

            types =
              pkgs.runCommand "persona-review-types"
                {
                  nativeBuildInputs = [
                    pkgs.basedpyright
                    # pythonEnv, not a bare interpreter: strict mode infers Unknown for an
                    # unresolvable `import pytest`, so tests/ would be checked against nothing.
                    pythonEnv
                  ];
                }
                ''
                  cp -r ${self}/. work && chmod -R u+w work
                  cd work
                  basedpyright

                  # Negative control, and the reason it exists: this check once passed while
                  # analysing ONE file, because `include` named a directory holding shell
                  # scripts. It reported strict and clean over none of the code that ships. A
                  # type check that cannot fail is worse than none, so prove it rejects a
                  # library it should before believing it about the real one.
                  cp -r ${self}/. broken && chmod -R u+w broken
                  cd broken
                  printf '\n\ndef _negative_control(x: int) -> str:\n    return x + None\n' \
                    >> persona_review/validate.py
                  if basedpyright > control.log 2>&1; then
                    echo "FAIL: basedpyright passed a library with \`return x + None\` in it." >&2
                    echo "The type check is not covering persona_review/. See pyrightconfig.json." >&2
                    exit 1
                  fi
                  grep -q 'reportOperatorIssue' control.log || {
                    echo "FAIL: basedpyright failed the control, but not on the injected error:" >&2
                    cat control.log >&2
                    exit 1
                  }
                  echo "ok: type check rejects a deliberately broken persona_review/validate.py"
                  touch $out
                '';

            lint =
              pkgs.runCommand "persona-review-lint"
                {
                  nativeBuildInputs = [ pkgs.ruff ];
                }
                ''
                  cp -r ${self}/. work && chmod -R u+w work
                  cd work
                  ruff check persona_review tests
                  ruff format --check persona_review tests
                  touch $out
                '';
          };
        }
      );
}
