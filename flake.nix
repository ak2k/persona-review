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
    flake-utils.lib.eachDefaultSystem (
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
            makeWrapper ${pkgs.python3}/bin/python3 $out/bin/ce-persona-findings \
              --add-flags "-m persona_review.findings" \
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
          wrappers = pkgs.runCommand "persona-review-wrappers" {
            nativeBuildInputs = [
              pkgs.python3
              pkgs.bash
              pkgs.coreutils
            ];
          } ''
            PERSONA_REVIEW_BIN=${persona-review}/bin \
              bash ${self}/tests/test-ce-persona-wrappers.sh
            touch $out
          '';

          unit = pkgs.runCommand "persona-review-unit" {
            nativeBuildInputs = [ pkgs.python3 ];
            PYTHONDONTWRITEBYTECODE = "1";
          } ''
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
            PYTHONPATH=${persona-review}/lib \
            PERSONA_REVIEW_BIN=${self}/bin \
              python3 ${self}/tests/test_ce_persona_validate.py
            touch $out
          '';

          # Strict from the first commit, with no baseline. See pyrightconfig.json.
          types = pkgs.runCommand "persona-review-types" {
            nativeBuildInputs = [
              pkgs.basedpyright
              pkgs.python3
            ];
          } ''
            cp -r ${self}/. work && chmod -R u+w work
            cd work
            basedpyright
            touch $out
          '';

          lint = pkgs.runCommand "persona-review-lint" {
            nativeBuildInputs = [
              pkgs.shellcheck
              pkgs.ruff
            ];
          } ''
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
