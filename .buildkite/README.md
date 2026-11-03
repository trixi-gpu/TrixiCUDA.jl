# Codecov coverage reports

Coverage is collected and uploaded by **Buildkite**, not GitHub Actions. This
follows [CUDA.jl's Buildkite pipeline](https://github.com/JuliaGPU/CUDA.jl/blob/main/.buildkite/pipeline.yml)
and uses the same [JuliaCI coverage plugin](https://github.com/JuliaCI/julia-coverage-buildkite-plugin).
`codecov.yml` configures the reports; it does not run tests or upload data.

## Collection and reports

The existing Julia 1.10 GPU test step runs `Pkg.test(; coverage = "user")`.
The Julia 1.11/1.12/1.13 steps continue to run without coverage instrumentation;
there is no additional test run. The coverage plugin runs after the test command,
merges Julia's `.cov` files under `src/` into `lcov.info`, and uploads that file
with the Codecov CLI. Buildkite also retains `lcov.info` as an artifact so that
it can be inspected even when submission fails. Other directories and dependency
sources are not included in this upload.

The plugin supplies the commit, branch, PR, and build metadata from Buildkite and
uses the GitHub repository slug. Both PR and main builds use this path, except
when `[skip tests]` skips the tests. Once main has a successful upload, Codecov
can compare the PR against its base. Initial comments may appear without a base
report, but cannot show a coverage change until both reports exist.

Overall and patch statuses are informational: coverage percentages do not fail
these checks. The PR comment is updated in place and includes overall coverage,
the diff, and affected files. Inline annotations are disabled. Upload errors do
not fail the test job (`fail-on-error: false`), but test failures still do.
An absent Codecov report or an error in the build log is not a successful upload.

## Account setup

Install the official Codecov GitHub App for `trixi-gpu` and grant it access to
`TrixiCUDA.jl`. For uploads, use either of these supported authentication setups:

- For this public repository, an organization administrator can allow tokenless
  uploads in Codecov's organization **Settings > Global Upload Token**, with
  token authentication for public repositories set to **Not required**. This is
  a Codecov account setting, not an option in `codecov.yml`.
- If token authentication is required, provide this repository's upload token as
  the `CODECOV_TOKEN` secret in the **Buildkite GPU job environment**. The plugin
  reads it automatically. A GitHub Actions secret alone does not reach Buildkite.
  The JuliaGPU pipeline administrator may need to configure secret injection.

Codecov's current [token documentation](https://docs.codecov.com/docs/codecov-tokens)
allows public repositories to disable token authentication with the modern CLI.
The plugin's README still describes requiring a token on Buildkite; the plugin
uses a modern CLI and passes `CODECOV_TOKEN` when it is supplied. An app installed
on GitHub does not by itself configure upload authentication.

To verify the setup, inspect the Julia 1.10 job's **Processing coverage** section,
check that `lcov.info` contains `SF:src/...` records and line counts, and confirm
that the corresponding commit appears on
[Codecov](https://app.codecov.io/gh/trixi-gpu/TrixiCUDA.jl). If the upload fails,
check the authentication setting and repository authorization before interpreting
an empty Codecov page as missing test coverage. Re-run a main build after merging
to establish the first baseline. This draft's GPU collection and live upload
must be validated on the Buildkite agent; configuration checks alone cannot
validate a real coverage report.

## Interpreting GPU coverage

Host-side coverage records executed Julia code. Device-side coverage has a
different meaning: GPUCompiler records source lines visited while compiling
tracked device code. The counts represent **compilations, not GPU executions**;
a covered kernel line is not proof that every GPU branch ran. This is documented
in [GPUCompiler's coverage implementation](https://github.com/JuliaGPU/GPUCompiler.jl/blob/main/src/jlgen.jl).

Use the report to identify code needing more tests, while retaining numerical
correctness tests and performance benchmarks as separate checks. Coverage
instrumentation must not be enabled when measuring benchmark performance.
