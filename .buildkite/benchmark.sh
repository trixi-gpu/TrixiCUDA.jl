#!/usr/bin/env bash
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
head_sha=$(git rev-parse HEAD)
results="$repo/benchmark/results"
mkdir -p "$results"
results=$(cd "$results" && pwd)
rm -f "$results/report.md" "$results/baseline.toml" "$results/target.toml"
work=$(mktemp -d)
cleanup() {
    git -C "$repo" worktree remove --force "$work/baseline" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT

default_branch="${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-main}"
base_branch=""
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
    base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:?Missing PR target branch}"
elif [[ "${BUILDKITE_BRANCH:-$default_branch}" != "$default_branch" ]]; then
    # Push builds can lack PR metadata; compare with the default branch.
    base_branch="$default_branch"
fi

if [[ -n "$base_branch" ]]; then
    git fetch --no-tags origin "$base_branch"
    base_sha=$(git rev-parse FETCH_HEAD)
else
    # First parent is the previous default-branch baseline, including merge commits.
    base_sha=$(git rev-parse "${head_sha}^1")
fi
printf 'Baseline: %s\nTarget: %s\n' "$base_sha" "$head_sha"
git worktree add --detach "$work/baseline" "$base_sha"

# Always run the target's harness against both source trees. This also bootstraps
# comparison against commits predating the benchmark infrastructure.
export JULIA_NUM_THREADS=1
julia --threads=1 "$repo/benchmark/ci/test_report.jl"
julia --threads=1 "$repo/benchmark/ci/prepare.jl" \
    "$repo/benchmark" "$repo" "$work/baseline" "$results/environments"
julia --threads=1 --project="$results/environments/target" "$repo/benchmark/ci/test_cases.jl"
for revision in baseline target; do
    sha="$base_sha"
    if [[ "$revision" == target ]]; then sha="$head_sha"; fi
    julia --threads=1 --project="$results/environments/$revision" \
        "$repo/benchmark/ci/run.jl" "$results/$revision.toml" "$sha"
done
julia --threads=1 "$repo/benchmark/ci/report.jl" \
    "$results/baseline.toml" "$results/target.toml" "$results/report.md"
if [[ -n "${BUILDKITE:-}" ]]; then
    buildkite-agent annotate --context gpu-rhs-benchmarks --style info < "$results/report.md"
fi
