// Buildkite's public annotation is the detailed report; no Buildkite API token
// or PR-controlled code is needed in this privileged GitHub status workflow.
const MARKER = '<!-- trixicuda-gpu-rhs-benchmarks -->';
const BUILD = /^https:\/\/buildkite\.com\/julialang\/trixicuda-dot-jl\/builds\/[1-9]\d*(?:#[\da-f-]+)?$/;
const SHA = /^[\da-f]{40}$/;

export function parseReport(html, target) {
    if (typeof html !== 'string' || html.length > 200_000 || !SHA.test(target))
        throw new Error('Invalid benchmark report');
    if (!html.includes('Minimum wall time with GPU synchronization'))
        throw new Error('Expected a minimum-time benchmark report');
    const baseline = html.match(/Baseline:\s*<code>([\da-f]{40})<\/code>/)?.[1];
    const measured = html.match(/Target:\s*<code>([\da-f]{40})<\/code>/)?.[1];
    const summary = html.match(/Cases: (\d+); geometric mean speedup: <strong>(\d+(?:\.\d+)?)x<\/strong>/);
    if (!baseline || measured !== target || !summary)
        throw new Error('Benchmark report does not match the measured commit');
    const rows = [];
    for (const [, row] of html.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)) {
        const cells = [...row.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((m) =>
            m[1].replace(/&#(?:x([\da-f]+)|(\d+));/gi, (_, hex, decimal) =>
                String.fromCodePoint(Number.parseInt(hex ?? decimal, hex ? 16 : 10))).trim());
        if (!cells.length) continue;
        if (cells.length !== 6 || !/^[a-z0-9_]{1,100}$/.test(cells[0]) || !/^\d+$/.test(cells[1]))
            throw new Error('Invalid benchmark row');
        const decimal = /^\d+(?:\.\d+)?$/;
        if (!decimal.test(cells[2]) || !decimal.test(cells[3]) ||
            !/^[+-]?\d+(?:\.\d+)?%$/.test(cells[4]) || !/^\d+(?:\.\d+)?x$/.test(cells[5]))
            throw new Error('Invalid benchmark timing');
        const change = Number(cells[4].slice(0, -1));
        const speedup = Number(cells[5].slice(0, -1));
        if (![Number(cells[2]), Number(cells[3])].every((n) => Number.isFinite(n) && n > 0) ||
            !Number.isFinite(speedup) || speedup < 0 ||
            !Number.isFinite(change) || change <= -100)
            throw new Error('Invalid benchmark timing');
        rows.push({name: cells[0], change, changeText: cells[4], speedupText: cells[5]});
    }
    const count = Number(summary[1]);
    const speedup = Number(summary[2]);
    if (rows.length !== count || count < 1 || count > 1000 ||
        new Set(rows.map((r) => r.name)).size !== count || !Number.isFinite(speedup) || speedup < 0)
        throw new Error('Invalid benchmark summary');
    return {baseline, target, rows, speedup: summary[2]};
}

export function formatComment(report, buildUrl) {
    // TODO: Agree on the final PR summary with maintainers. For now, show the
    // geometric mean and the three largest absolute time changes, with no
    // significance claims; the annotation keeps every case and environment.
    const rows = [...report.rows].sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 3);
    return [
        MARKER, '### GPU RHS benchmarks', '',
        `\`${report.target.slice(0, 7)}\` vs merge base \`${report.baseline.slice(0, 7)}\`; ` +
            `${report.rows.length} cases, minimum synchronized wall time.`,
        `Geometric mean speedup: **${report.speedup}x**.`, '',
        '| Largest time changes | Time change | Speedup |',
        '|:--|--:|--:|',
        ...rows.map((r) => `| \`${r.name}\` | ${r.changeText} | ${r.speedupText} |`), '',
        `[Full report and artifacts](${buildUrl}). Informational; differences may be measurement noise.`,
    ].join('\n');
}

export async function publish({github, context, fetchImpl = fetch, log = console.log}) {
    const {sha, target_url: targetUrl} = context.payload;
    if (!SHA.test(sha) || typeof targetUrl !== 'string' || !BUILD.test(targetUrl))
        throw new Error('Unexpected Buildkite status');
    const buildUrl = targetUrl.split('#')[0];
    async function readJson(path) {
        const response = await fetchImpl(buildUrl + path, {
            headers: {Accept: 'application/json'}, redirect: 'error', signal: AbortSignal.timeout(30_000),
        });
        if (!response.ok) throw new Error(`Buildkite request failed: ${response.status}`);
        return response.json();
    }
    const build = await readJson('.json');
    if (build.commit_id !== sha) throw new Error('Build does not match the status commit');
    if (!build.pull_request) return log('No PR: the report remains in the annotation and artifacts.');
    if (!/^[1-9]\d*$/.test(String(build.pull_request.id))) throw new Error('Invalid pull request number');
    const pull_number = Number(build.pull_request.id);
    const repo = context.repo;
    const currentPR = async () => (await github.rest.pulls.get({...repo, pull_number})).data;
    const eligible = (pr) => pr.state === 'open' && !pr.draft && pr.head.sha === sha;
    if (!eligible(await currentPR())) return log('The PR is closed, draft, or has a newer head.');
    const steps = await readJson('/data/steps');
    if (!Array.isArray(steps) || !steps.some((s) => s.key === 'benchmarks' && s.outcome === 'passed'))
        return log('No successful benchmark step.');
    const annotations = await readJson('/data/annotations');
    const report = annotations.records?.find((a) => a.context === 'gpu-rhs-benchmarks');
    if (!report) return log('No benchmark annotation.');
    const body = formatComment(parseReport(report.body?.html, sha), buildUrl);
    const comments = await github.paginate(github.rest.issues.listComments, {...repo, issue_number: pull_number, per_page: 100});
    const existing = comments.filter((c) => c.user?.login === 'github-actions[bot]' && c.body?.startsWith(MARKER)).at(-1);
    // Recheck after reading the report so a completed old build does not replace
    // the comment of a head pushed while this workflow was running.
    if (!eligible(await currentPR())) return log('The PR moved on while publishing.');
    if (existing) {
        if (existing.body !== body)
            await github.rest.issues.updateComment({...repo, comment_id: existing.id, body});
    } else {
        await github.rest.issues.createComment({...repo, issue_number: pull_number, body});
    }
}
