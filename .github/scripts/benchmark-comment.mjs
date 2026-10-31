// Buildkite's public annotation is the detailed report; no Buildkite API token
// is needed. The status workflow runs trusted code from the default branch.
const MARKER = '<!-- trixicuda-gpu-rhs-benchmarks -->';
const BRIDGE = '_Posted from the completed Buildkite report as a pre-merge bridge;';
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
    const finished = ['passed', 'failed', 'canceled', 'not_run'].includes(build.state);
    if (!build.pull_request) {
        log('No PR: the report remains in the annotation and artifacts.');
        return 'skipped';
    }
    if (!/^[1-9]\d*$/.test(String(build.pull_request.id))) throw new Error('Invalid pull request number');
    const pull_number = Number(build.pull_request.id);
    const repo = context.repo;
    const currentPR = async () => (await github.rest.pulls.get({...repo, pull_number})).data;
    const eligible = (pr) => pr.state === 'open' && !pr.draft && pr.head.sha === sha;
    const pr = await currentPR();
    if (!eligible(pr)) {
        log('The PR is closed, draft, or has a newer head.');
        return 'skipped';
    }
    const steps = await readJson('/data/steps');
    if (!Array.isArray(steps)) throw new Error('Invalid Buildkite steps');
    const step = steps.find((s) => s.key === 'benchmarks');
    if (step?.outcome !== 'passed') {
        if (step?.outcome || finished) {
            log('No successful benchmark step.');
            return 'skipped';
        }
        return 'pending';
    }
    const annotations = await readJson('/data/annotations');
    const report = annotations.records?.find((a) => a.context === 'gpu-rhs-benchmarks');
    if (!report) {
        if (finished) throw new Error('Successful benchmark has no annotation');
        return 'pending';
    }
    const body = formatComment(parseReport(report.body?.html, sha), buildUrl);
    const comments = await github.paginate(github.rest.issues.listComments, {...repo, issue_number: pull_number, per_page: 100});
    const existing = comments.filter((c) => c.user?.login === 'github-actions[bot]' && c.body?.startsWith(MARKER)).at(-1);
    // Recheck after reading the report so a completed old build does not replace
    // the comment of a head pushed while this workflow was running.
    if (!eligible(await currentPR())) {
        log('The PR moved on while publishing.');
        return 'skipped';
    }
    if (existing) {
        if (existing.body !== body)
            await github.rest.issues.updateComment({...repo, comment_id: existing.id, body});
    } else {
        await github.rest.issues.createComment({...repo, issue_number: pull_number, body});
    }
    // Replace only the explicitly marked, author-posted transitional report,
    // after the Actions bot has successfully published its own comment.
    for (const comment of comments) {
        if (pr.user?.login && comment.user?.login === pr.user.login &&
            comment.body?.startsWith(MARKER) && comment.body.includes(BRIDGE))
            await github.rest.issues.deleteComment({...repo, comment_id: comment.id});
    }
    return 'published';
}

export async function publishPullRequest({github, context, fetchImpl = fetch, log = console.log,
    sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), now = Date.now,
    timeoutMs = 165 * 60_000, intervalMs = 60_000}) {
    const eventPR = context.payload.pull_request;
    const sha = eventPR?.head?.sha;
    if (!SHA.test(sha) || !Number.isSafeInteger(eventPR?.number) || eventPR.number < 1)
        throw new Error('Invalid pull request event');
    const repo = context.repo;
    const eligible = (pr) => pr.state === 'open' && !pr.draft && pr.head.sha === sha &&
        pr.head.repo?.full_name === `${repo.owner}/${repo.repo}`;
    if (!eligible(eventPR)) return 'skipped';
    // Once this workflow is on the default branch, status events publish reports
    // without keeping a runner waiting. This path bootstraps same-repository PRs.
    try {
        await github.rest.repos.getContent({...repo, path: '.github/workflows/Benchmark.yml',
            ref: context.payload.repository.default_branch});
        log('The default-branch status workflow will publish this report.');
        return 'skipped';
    } catch (error) {
        if (error.status !== 404) throw error;
    }
    const commit = (await github.rest.repos.getCommit({...repo, ref: sha})).data;
    if (commit.commit.message.includes('[skip benchmarks]')) return 'skipped';
    const deadline = now() + timeoutMs;
    log('Waiting for the current head\'s Buildkite benchmark report.');
    while (true) {
        const pr = (await github.rest.pulls.get({...repo, pull_number: eventPR.number})).data;
        if (!eligible(pr)) return 'skipped';
        const combined = (await github.rest.repos.getCombinedStatusForRef({...repo, ref: sha})).data;
        const status = combined.statuses.find((s) => s.context === 'buildkite/trixicuda-dot-jl');
        if (status) {
            const result = await publish({github, fetchImpl, log, context: {repo,
                payload: {sha, target_url: status.target_url}}});
            if (result !== 'pending') return result;
        }
        if (now() >= deadline) throw new Error('Timed out waiting for the benchmark report');
        await sleep(Math.min(intervalMs, deadline - now()));
    }
}
