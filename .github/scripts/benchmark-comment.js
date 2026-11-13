module.exports = async ({github, context}) => {
  const fs = require('fs');
  const repo = context.repo, number = Number(process.env.PR_NUMBER);
  const sha = process.env.HEAD_SHA, baseline = process.env.BASELINE_SHA;
  const runURL = `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;

  const read = file => JSON.parse(fs.readFileSync(`results/${file}`, 'utf8'));
  const base = read('baseline.json'), target = read('target.json');
  const rows = Object.entries(target.benchmarks).map(([name, t]) => {
    const b = base.benchmarks[name];
    return {
      name,
      speedup: b.minimum_ns / t.minimum_ns,
      change: 100 * (t.minimum_ns / b.minimum_ns - 1)
    };
  });
  const speedup = Math.exp(rows.reduce((sum, r) => sum + Math.log(r.speedup), 0) / rows.length);

  const marker = '<!-- trixicuda-gpu-rhs-benchmarks -->';
  const body = [marker, '### GPU RHS Benchmarks', '',
    `\`${sha.slice(0, 7)}\` vs merge base \`${baseline.slice(0, 7)}\`; ` +
      `${rows.length} cases, minimum synchronized wall time.`,
    `Geometric mean speedup: **${speedup.toFixed(3)}x**.`, '',
    '| Largest time changes | Time change | Speedup |', '|:--|--:|--:|',
    ...rows.sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 3).map(r =>
      `| \`${r.name}\` | ${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)}% | ` +
      `${r.speedup.toFixed(3)}x |`), '',
    `[Full benchmark report](${runURL})`,
    'Results are informational only; differences may be measurement noise.'].join('\n');

  // Find this bot's marked comment so repeated runs update the same discussion.
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo, issue_number: number, per_page: 100
  });
  const existing = comments.find(c =>
    c.user?.login === 'github-actions[bot]' && c.body?.startsWith(marker));

  // Recheck the PR before posting because it may have changed during the run.
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  if (pr.state !== 'open' || pr.draft || pr.head.sha !== sha ||
      pr.base.ref !== process.env.BASE_BRANCH) return;
  if (existing) {
    await github.rest.issues.updateComment({...repo, comment_id: existing.id, body});
  } else {
    await github.rest.issues.createComment({...repo, issue_number: number, body});
  }
};
