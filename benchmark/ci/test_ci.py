"""Exercise the CI shell driver against real Git histories without requiring a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


DRIVER = Path(__file__).resolve().parents[2] / ".buildkite" / "benchmark.sh"


class BaselineSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "origin.git"
        self.repo = self.root / "checkout"
        subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(self.remote)],
                       check=True, capture_output=True)
        self.repo.mkdir()
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Benchmark tests")
        self.git("config", "user.email", "benchmark-tests@example.invalid")
        self.git("remote", "add", "origin", str(self.remote))
        (self.repo / ".buildkite").mkdir()
        shutil.copyfile(DRIVER, self.repo / ".buildkite" / "benchmark.sh")
        self.initial = self.commit("initial.txt")
        self.git("push", "origin", "main")
        self.git("checkout", "-b", "feature")
        self.commit("first.txt")
        self.commit("second.txt")
        self.git("checkout", "main")
        self.main = self.commit("main.txt")
        self.git("push", "origin", "main")
        self.git("checkout", "feature")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "julia.jsonl"
        stub = self.bin / "julia"
        stub.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['BENCHMARK_TEST_LOG'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if any(a.endswith('/ci/report.jl') for a in args):
    pathlib.Path(args[-1]).write_text('test report')
""")
        stub.chmod(0o755)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              text=True, capture_output=True).stdout.strip()

    def commit(self, filename):
        (self.repo / filename).touch()
        self.git("add", ".")
        self.git("commit", "-m", filename)
        return self.git("rev-parse", "HEAD")

    def run_driver(self, expected, branch="feature", pr=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith("BUILDKITE")}
        env.update(PATH=str(self.bin) + os.pathsep + env["PATH"],
                   BENCHMARK_TEST_LOG=str(self.log), BUILDKITE_BRANCH=branch,
                   BUILDKITE_PIPELINE_DEFAULT_BRANCH="main", BUILDKITE_PULL_REQUEST="227" if pr else "false")
        if pr:
            env["BUILDKITE_PULL_REQUEST_BASE_BRANCH"] = "main"
        result = subprocess.run(["bash", ".buildkite/benchmark.sh"], cwd=self.repo,
                                env=env, capture_output=True, text=True, check=True)
        head = self.git("rev-parse", "HEAD")
        self.assertIn(f"Baseline: {expected}\nTarget: {head}", result.stdout)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        runs = [args for args in calls if any(a.endswith("/ci/run.jl") for a in args)]
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0][-1], head)
        self.assertEqual(runs[1][-2:], [expected, "--reuse"])
        self.assertTrue(runs[0][-2].endswith("/results/target.json"))
        self.assertTrue(runs[1][-3].endswith("/results/baseline.json"))
        for script in ("test_report.jl", "test_cache.jl", "report.jl"):
            args = next(a for a in calls if any(v.endswith("/ci/" + script) for v in a))
            self.assertTrue(any(v.startswith("--project=") and
                                v.endswith("/environments/target") for v in args))
        report = next(a for a in calls if any(v.endswith("/ci/report.jl") for v in a))
        self.assertTrue(report[-3].endswith("/baseline.json"))
        self.assertTrue(report[-2].endswith("/target.json"))
        self.assertEqual(self.git("worktree", "list", "--porcelain").count("worktree "), 1)

    def test_pr_uses_merge_base_and_only_measures_pushed_head(self):
        self.run_driver(self.initial, pr=True)

    def test_branch_push_without_pr_metadata_uses_merge_base(self):
        self.run_driver(self.initial)

    def test_merging_main_advances_merge_base(self):
        self.git("merge", "--no-edit", "main")
        self.run_driver(self.main, pr=True)

    def test_main_merge_build_uses_first_parent(self):
        self.git("checkout", "main")
        self.git("merge", "--no-ff", "--no-edit", "feature")
        self.run_driver(self.main, branch="main")

    def test_main_push_with_several_commits_uses_last_first_parent(self):
        self.git("checkout", "main")
        previous = self.commit("main-second.txt")
        self.commit("main-third.txt")
        self.run_driver(previous, branch="main")

    def test_shallow_clone_recovers_history_for_merge_base(self):
        self.git("push", "origin", "feature")
        shallow = self.root / "shallow"
        subprocess.run(["git", "clone", "--depth=1", "--branch=feature",
                        self.remote.as_uri(), str(shallow)], check=True, capture_output=True)
        self.repo = shallow
        self.run_driver(self.initial, pr=True)


if __name__ == "__main__":
    unittest.main()
