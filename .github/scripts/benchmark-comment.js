module.exports = async ({github, context}) => {
  const fs = require('fs');
  const repo = context.repo, number = Number(process.env.PR_NUMBER);
  const sha = process.env.HEAD_SHA, baseline = process.env.BASELINE_SHA;
  const runURL = `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;
  const eligible = pr => pr.state === 'open' && !pr.draft && pr.head.sha === sha &&
    pr.head.repo?.full_name === `${repo.owner}/${repo.repo}` && pr.base.ref === process.env.BASE_BRANCH;
  const getPR = async () => (await github.rest.pulls.get({...repo, pull_number: number})).data;
  if (!eligible(await getPR())) return;
  const base = JSON.parse(fs.readFileSync('results/baseline.json', 'utf8'));
  const target = JSON.parse(fs.readFileSync('results/target.json', 'utf8'));
  const report = fs.readFileSync('results/report.md', 'utf8');
  if (!/^[a-f0-9]{40}$/.test(sha) || !/^[a-f0-9]{40}$/.test(baseline) ||
      base.schema !== 3 || target.schema !== 3 || target.reused !== false ||
      base.environment?.commit !== baseline || target.environment?.commit !== sha ||
      target.environment.build_url !== runURL || !report.includes(sha) || !report.includes(baseline))
    throw new Error('Report does not belong to this comparison');
  // The Julia report writer already validates environments, inputs and raw samples.
  const valid = value => Number.isFinite(value) && value > 0;
  const rows = Object.entries(target.benchmarks).map(([name, t]) => {
    const b = base.benchmarks[name];
    if (!/^[a-z0-9_]{1,100}$/.test(name) || !valid(b?.minimum_ns) || !valid(t.minimum_ns))
      throw new Error('Invalid benchmark timing');
    return {name, speedup: b.minimum_ns / t.minimum_ns, change: 100 * (t.minimum_ns / b.minimum_ns - 1)};
  });
  const speedup = Math.exp(rows.reduce((sum, r) => sum + Math.log(r.speedup), 0) / rows.length);
  if (!valid(speedup) || !rows.every(r => valid(r.speedup) && Number.isFinite(r.change)))
    throw new Error('Invalid speedup');
  const marker = '<!-- trixicuda-gpu-rhs-benchmarks -->';
  // TODO: Agree on the final summary format; keep all cases in the job summary.
  const body = [marker, '### GPU RHS benchmarks', '',
    `\`${sha.slice(0, 7)}\` vs merge base \`${baseline.slice(0, 7)}\`; ${rows.length} cases, minimum synchronized wall time.`,
    `Geometric mean speedup: **${speedup.toFixed(3)}x**.`, '',
    '| Largest time changes | Time change | Speedup |', '|:--|--:|--:|',
    ...rows.sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 3).map(r =>
      `| \`${r.name}\` | ${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)}% | ${r.speedup.toFixed(3)}x |`), '',
    `[Full benchmark report](${runURL})`,
    'Informational; differences may be measurement noise.'].join('\n');
  const comments = await github.paginate(github.rest.issues.listComments, {...repo, issue_number: number, per_page: 100});
  const existing = comments.find(c => c.user?.login === 'github-actions[bot]' && c.body?.startsWith(marker));
  if (!eligible(await getPR())) return;
  if (existing) {
    if (existing.body !== body) await github.rest.issues.updateComment({...repo, comment_id: existing.id, body});
  } else {
    await github.rest.issues.createComment({...repo, issue_number: number, body});
  }
};
