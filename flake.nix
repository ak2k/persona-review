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
    # `nix flake check --all-systems` failed on evaluation alone rather than on anything
    # in this repo. Declare what is actually supported.
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

          # `grok` and `codex` are deliberately NOT dependencies. They are the user's own
          # subscription-authenticated CLIs, resolved from PATH at run time so a review uses
          # the login they already have rather than an API key this package would need.
          persona-review = pkgs.stdenvNoCC.mkDerivation {
            pname = "persona-review";
            version = "0.1.0";
            src = ./.;

            nativeBuildInputs = [ pkgs.makeWrapper ];

            installPhase = ''
              runHook preInstall

              mkdir -p $out/bin $out/lib
              cp -r persona_review $out/lib/
              cp bin/ce-grok-persona bin/ce-codex-persona $out/bin/
              chmod +x $out/bin/*

              # The findings projection is a caller-facing command, not just a module.
              # PYTHONSAFEPATH because `python3 -m` puts the CALLER'S cwd first on sys.path,
              # ahead of PYTHONPATH — so without it a `persona_review/` directory in
              # whatever repo the caller happens to be standing in replaces this module.
              makeWrapper ${pkgs.python3}/bin/python3 $out/bin/ce-persona-findings \
                --add-flags "-m persona_review.findings" \
                --set PYTHONSAFEPATH 1 \
                --set PYTHONPATH $out/lib

              # The wrappers locate the gate by importing `persona_review`, which needs a
              # PYTHONPATH root. Binding it here is what makes the packaged build honest:
              # $out/bin holds only entry points, so the source-checkout fallback (the
              # directory above the wrapper) would find nothing.
              for w in ce-grok-persona ce-codex-persona; do
                wrapProgram $out/bin/$w \
                  --set PERSONA_REVIEW_PYTHONPATH $out/lib \
                  --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.python3 ]}
              done

              runHook postInstall
            '';

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
              pkgs.python3
              pkgs.shellcheck
              pkgs.ruff
              pkgs.basedpyright
            ];
            shellHook = ''
              export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };

          checks = {
            # Runs the packaged wrappers as processes against stub runners: exit status
            # through the pipeline, refusals before dispatch, and the one-line stdout
            # contract. This is the reason the code lives in its own repo — a check that
            # reads ./bin proves the files are fine, while this proves what actually ships
            # is, wrapper substitution and patched shebangs included.
            wrappers =
              pkgs.runCommand "persona-review-wrappers"
                {
                  nativeBuildInputs = [
                    pkgs.python3
                    pkgs.bash
                    pkgs.coreutils
                    # For the budget test: the size preflight measures `git diff BASE..HEAD`,
                    # so proving it fires needs a real repository with a real large diff.
                    pkgs.git
                  ];
                }
                ''
                  PERSONA_REVIEW_BIN=${persona-review}/bin \
                    bash ${self}/tests/test-ce-persona-wrappers.sh
                  touch $out
                '';

            unit =
              pkgs.runCommand "persona-review-unit"
                {
                  nativeBuildInputs = [ pkgs.python3 ];
                  PYTHONDONTWRITEBYTECODE = "1";
                }
                ''
                  # Packaged library, SOURCE wrappers — and the split is deliberate.
                  #
                  # The Python under test is the packaged copy, so the modules that ship are
                  # the modules exercised. But the wrapper invariants assert the shape of the
                  # authored script (the gate is the last command, unguarded; --json-schema is
                  # requested; no transcript on stdout), and $out/bin holds makeWrapper shims
                  # — `exec -a "$0" …-wrapped` — which carry none of that text. Pointing these
                  # assertions at the packaged bin makes them assert the wrapper generator
                  # instead of our code, which is how a guard quietly stops guarding.
                  #
                  # The packaged wrappers are covered, as processes, by the `wrappers` check.
                  # PERSONA_REVIEW_EXPECT_LIB makes the first half of that claim testable, and
                  # it did not hold: the suite put its own source root at sys.path[0], which
                  # outranks PYTHONPATH, so it imported the tree beside it and passed with every
                  # packaged module replaced by `raise RuntimeError`. The suite now asserts what
                  # it actually imported, because only that process knows.
                  PYTHONPATH=${persona-review}/lib \
                  PERSONA_REVIEW_EXPECT_LIB=${persona-review}/lib \
                  PERSONA_REVIEW_BIN=${self}/bin \
                    python3 ${self}/tests/test_ce_persona_validate.py
                  touch $out
                '';

            # Strict from the first commit, with no baseline. See pyrightconfig.json.
            types =
              pkgs.runCommand "persona-review-types"
                {
                  nativeBuildInputs = [
                    pkgs.basedpyright
                    pkgs.python3
                  ];
                }
                ''
                  cp -r ${self}/. work && chmod -R u+w work
                  cd work
                  basedpyright

                  # Negative control, and the reason it exists: this check once passed while
                  # analysing ONE file — the test module — because `include` named bin/, which
                  # holds shell scripts. It reported strict and clean over none of the code that
                  # ships. A type check that cannot fail is worse than none, so prove it fails
                  # on a library the checker should reject before believing it about the real one.
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
                    echo "FAIL: basedpyright failed the control, but not on the injected type error:" >&2
                    cat control.log >&2
                    exit 1
                  }
                  echo "ok: type check rejects a deliberately broken persona_review/validate.py"
                  touch $out
                '';

            lint =
              pkgs.runCommand "persona-review-lint"
                {
                  nativeBuildInputs = [
                    pkgs.shellcheck
                    pkgs.ruff
                  ];
                }
                ''
                  cp -r ${self}/. work && chmod -R u+w work
                  cd work
                  shellcheck bin/ce-grok-persona bin/ce-codex-persona tests/test-ce-persona-wrappers.sh
                  ruff check persona_review tests
                  ruff format --check persona_review tests
                  touch $out
                '';
          };
        }
      );
}
