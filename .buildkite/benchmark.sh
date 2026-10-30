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
branch="${BUILDKITE_BRANCH:-$(git branch --show-current)}"
if [[ $(git rev-parse --is-shallow-repository) == true ]]; then
    git fetch --unshallow origin
fi
if [[ "$branch" != "$default_branch" ]]; then
    base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-$default_branch}"
    git fetch --no-tags origin "$base_branch"
    base_sha=$(git merge-base "$head_sha" FETCH_HEAD)
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
julia --threads=1 "$repo/benchmark/ci/test_cache.jl"
julia --threads=1 "$repo/benchmark/ci/prepare.jl" \
    "$repo/benchmark" "$repo" "$work/baseline" "$results/environments"
julia --threads=1 --project="$results/environments/target" "$repo/benchmark/ci/test_cases.jl"
# Always measure the new head. Reuse the exact baseline only when its measurement
# context matches; otherwise run it using this target's harness and environment.
julia --threads=1 --project="$results/environments/target" \
    "$repo/benchmark/ci/run.jl" "$results/target.toml" "$head_sha"
julia --threads=1 --project="$results/environments/baseline" \
    "$repo/benchmark/ci/run.jl" "$results/baseline.toml" "$base_sha" --reuse
julia --threads=1 "$repo/benchmark/ci/report.jl" \
    "$results/baseline.toml" "$results/target.toml" "$results/report.md"
if [[ -n "${BUILDKITE:-}" ]]; then
    buildkite-agent annotate --context gpu-rhs-benchmarks --style info < "$results/report.md"
fi
