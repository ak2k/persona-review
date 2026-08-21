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

          # `grok` and `codex` are deliberately NOT dependencies. They are the user's own
          # subscription-authenticated CLIs, resolved from PATH at run time so a review uses
          # the login they already have rather than an API key this package would need.
          persona-review = python.pkgs.buildPythonApplication {
            pname = "persona-review";
            version = "0.2.0";
            pyproject = true;
            src = ./.;

            build-system = [ python.pkgs.setuptools ];

            # No test phase here: the suites need a writable HOME and stub runners on PATH,
            # and they are run as their own checks below against THIS built output.
            doCheck = false;

            # Belt and braces. A generated console script already puts its own directory on
            # sys.path rather than the caller's cwd, which is what closes the gate-replacement
            # hole; this also covers anyone who reaches for `python -m persona_review.…`.
            makeWrapperArgs = [
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
              python
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
            process = pkgs.runCommand "persona-review-process" {
              nativeBuildInputs = [
                python
                pkgs.coreutils
                # The size preflight measures `git diff BASE..HEAD`, so proving it fires
                # needs a real repository with a real large diff.
                pkgs.git
              ];
            } ''
              export HOME=$(mktemp -d)
              PERSONA_REVIEW_BIN=${persona-review}/bin \
                python3 ${self}/tests/test_process.py
              touch $out
            '';

            unit = pkgs.runCommand "persona-review-unit" {
              nativeBuildInputs = [ python ];
              PYTHONDONTWRITEBYTECODE = "1";
            } ''
              export HOME=$(mktemp -d)
              # The library under test is the PACKAGED copy, so the modules that ship are the
              # modules exercised. PERSONA_REVIEW_EXPECT_LIB makes that claim testable from
              # inside the process, because only the process knows what it actually imported:
              # an earlier suite put its own source root ahead of PYTHONPATH and passed with
              # every packaged module replaced by `raise RuntimeError`.
              PYTHONPATH=$(echo ${persona-review}/${python.sitePackages}) \
              PERSONA_REVIEW_EXPECT_LIB=${persona-review} \
                python3 ${self}/tests/test_unit.py
              touch $out
            '';

            types = pkgs.runCommand "persona-review-types" {
              nativeBuildInputs = [
                pkgs.basedpyright
                python
              ];
            } ''
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

            lint = pkgs.runCommand "persona-review-lint" {
              nativeBuildInputs = [ pkgs.ruff ];
            } ''
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
