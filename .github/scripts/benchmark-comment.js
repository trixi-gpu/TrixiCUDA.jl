module.exports = async ({github, context}) => {
  const repo = context.repo, number = Number(process.env.PR_NUMBER);
  const sha = process.env.HEAD_SHA;
  const runURL = `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;

  const previewURL = `https://${repo.owner}.github.io/${repo.repo}/previews/PR${number}/benchmark/`;
  const {marker, text} = process.env.COMMENT_KIND === 'sweep' ? {
    marker: '<!-- trixicuda-gpu-rhs-sweep -->',
    text: `GPU RHS benchmark sweep for \`${sha.slice(0, 7)}\`: [heatmaps](${previewURL})`
  } : {
    marker: '<!-- trixicuda-gpu-rhs-benchmarks -->',
    text: `GPU RHS mini benchmark for \`${sha.slice(0, 7)}\`: [full report](${runURL})`
  };
  const body = `${marker}\n${text}`;

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
