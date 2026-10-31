import assert from 'node:assert/strict';
import test from 'node:test';
import {formatComment, parseReport, publish, publishPullRequest} from './benchmark-comment.mjs';

const target = 'a'.repeat(40);
const baseline = 'b'.repeat(40);
const buildUrl = 'https://buildkite.com/julialang/trixicuda-dot-jl/builds/313';
const html = `<p>Baseline: <code>${baseline}</code><br>Target: <code>${target}</code></p>
<p>Minimum wall time with GPU synchronization.</p>
<p>Cases: 4; geometric mean speedup: <strong>0.998x</strong>.</p>
<table><thead><tr><th>Case</th></tr></thead><tbody>
<tr><td>unchanged</td><td>32</td><td>1.0000</td><td>1.0000</td><td>+0.00%</td><td>1.000x</td></tr>
<tr><td>faster</td><td>32</td><td>2.0000</td><td>1.0000</td><td>-50.00%</td><td>2.000x</td></tr>
<tr><td>slower</td><td>32</td><td>1.0000</td><td>2.0000</td><td>+100.00%</td><td>0.500x</td></tr>
<tr><td>small_change</td><td>32</td><td>1.0000</td><td>1.0100</td><td>+1.00%</td><td>0.990x</td></tr>
</tbody></table>`;

test('minimum report becomes a compact comment with explicit speedups', () => {
    const report = parseReport(html, target);
    assert.equal(report.baseline, baseline);
    const body = formatComment(report, buildUrl);
    assert.ok(body.startsWith('<!-- trixicuda-gpu-rhs-benchmarks -->'));
    assert.ok(body.includes('4 cases, minimum synchronized wall time'));
    assert.ok(body.includes('Geometric mean speedup: **0.998x**'));
    assert.ok(body.includes('| `faster` | -50.00% | 2.000x |'));
    assert.ok(body.includes('| `slower` | +100.00% | 0.500x |'));
    assert.ok(!body.includes('| `unchanged` |'));
    assert.ok(body.includes(buildUrl));
});

test('accept HTML-escaped numeric punctuation and rounded tiny speedups', () => {
    const escaped = html.replaceAll('+', '&#43;').replaceAll('%', '&#37;');
    assert.equal(parseReport(escaped, target).rows.length, 4);
    assert.equal(parseReport(html.replace('0.500x', '0.000x'), target).rows.length, 4);
});

test('reject mismatched, old, malformed and injected reports', () => {
    assert.throws(() => parseReport(html, baseline));
    assert.throws(() => parseReport(html.replace('Minimum wall time', 'Median wall time'), target));
    assert.throws(() => parseReport(html.replace('Cases: 4', 'Cases: 5'), target));
    assert.throws(() => parseReport(html.replace('<td>faster</td>', '<td><script>alert(1)</script></td>'), target));
    assert.throws(() => parseReport(html.replace('2.000x', 'NaNx'), target));
    assert.throws(() => parseReport(html.replace('-50.00%', '-100.00%'), target));
    assert.throws(() => parseReport(html.replace('<td>faster</td>', '<td>unchanged</td>'), target));
    assert.throws(() => parseReport(null, target));
});

function fixture(options = {}) {
    const calls = [];
    let reads = 0;
    let statusReads = 0;
    let buildReads = 0;
    let clock = 0;
    const makePR = () => ({number: 227, user: {login: 'author'},
        state: options.closed ? 'closed' : 'open', draft: options.draft ?? false,
        head: {sha: options.stale || (options.movesOn && reads++ > 0) ? baseline : target,
            repo: {full_name: options.fork ? 'author/TrixiCUDA.jl' : 'trixi-gpu/TrixiCUDA.jl'}},
    });
    const github = {
        rest: {
            pulls: {get: async () => ({data: makePR()})},
            repos: {
                getContent: async () => {
                    if (options.defaultPublisher) return {data: {type: 'file'}};
                    throw Object.assign(new Error('Not on the default branch'), {status: 404});
                },
                getCommit: async () => ({data: {commit: {
                    message: options.skip ? 'change [skip benchmarks]' : 'change',
                }}}),
                getCombinedStatusForRef: async () => ({data: {statuses:
                    options.noStatus || (options.delayed && statusReads++ === 0) ? [] : [{
                        context: 'buildkite/trixicuda-dot-jl', target_url: buildUrl,
                    }],
                }}),
            },
            issues: {
                listComments: 'listComments',
                createComment: async (args) => calls.push({operation: 'create', ...args}),
                updateComment: async (args) => calls.push({operation: 'update', ...args}),
                deleteComment: async (args) => calls.push({operation: 'delete', ...args}),
            },
        },
        paginate: async () => options.comments ?? (options.existing ? [{id: 123, user: {login: 'github-actions[bot]'},
            body: '<!-- trixicuda-gpu-rhs-benchmarks -->\nold'}] : []),
    };
    const fetched = [];
    const fetchImpl = async (url) => {
        fetched.push(url);
        let data;
        if (url.endsWith('.json')) data = {commit_id: options.wrongCommit ? baseline : target,
            state: options.running && buildReads++ === 0 ? 'started' : 'passed',
            pull_request: options.main ? null : {id: '227'}};
        else if (url.endsWith('/steps')) data = [{key: 'benchmarks', outcome:
            options.running && buildReads === 1 ? null : options.failed ? 'soft_failed' : 'passed'}];
        else if (url.endsWith('/annotations')) data = {records: options.missing ? [] : [{context: 'gpu-rhs-benchmarks', body: {html}}]};
        else throw new Error(`Unexpected URL: ${url}`);
        return {ok: true, json: async () => data};
    };
    const context = {repo: {owner: 'trixi-gpu', repo: 'TrixiCUDA.jl'}, payload: {
        sha: target, target_url: options.url ?? buildUrl,
        pull_request: makePR(), repository: {default_branch: 'main'},
    }};
    context.payload.pull_request.head.sha = target;
    // Do not count the event snapshot as a live PR read in race-condition tests.
    reads = 0;
    return {args: {github, context, fetchImpl, log: () => {},
        now: () => clock, sleep: async (ms) => {clock += ms;}, timeoutMs: 120_000}, calls, fetched};
}

test('post once or update the existing bot comment', async () => {
    for (const existing of [false, true]) {
        const f = fixture({existing});
        await publish(f.args);
        assert.equal(f.calls.length, 1);
        assert.equal(f.calls[0].operation, existing ? 'update' : 'create');
        assert.equal(f.calls[0].issue_number ?? 227, 227);
        assert.ok(f.calls[0].body.includes('minimum synchronized wall time'));
    }
});

test('no comment for main, drafts, merged PRs, stale heads or failed benchmarks', async () => {
    for (const option of ['main', 'draft', 'closed', 'stale', 'movesOn', 'failed']) {
        const f = fixture({[option]: true});
        await publish(f.args);
        assert.equal(f.calls.length, 0, option);
        if (option === 'main') assert.equal(f.fetched.length, 1);
    }
});

test('a completed benchmark without its annotation is an error', async () => {
    const f = fixture({missing: true});
    await assert.rejects(publish(f.args), /no annotation/);
    assert.equal(f.calls.length, 0);
});

test('reject unrelated Buildkite URLs and commit mismatches', async () => {
    const wrongURL = fixture({url: 'https://example.com/julialang/trixicuda-dot-jl/builds/313'});
    await assert.rejects(publish(wrongURL.args));
    assert.equal(wrongURL.fetched.length, 0);
    const wrongCommit = fixture({wrongCommit: true});
    await assert.rejects(publish(wrongCommit.args));
    assert.equal(wrongCommit.calls.length, 0);
});

test('publish from a same-repository PR before the workflow reaches the default branch', async () => {
    const f = fixture({delayed: true, running: true});
    // Actions' context.repo is a getter, so spreading context would lose it.
    const repo = f.args.context.repo;
    delete f.args.context.repo;
    Object.defineProperty(f.args.context, 'repo', {get: () => repo});
    assert.equal(await publishPullRequest(f.args), 'published');
    assert.equal(f.calls.length, 1);
    assert.equal(f.calls[0].operation, 'create');
    assert.equal(f.fetched.filter((url) => url.endsWith('.json')).length, 2);
});

test('defer to the default-branch status workflow when it is available', async () => {
    const f = fixture({defaultPublisher: true});
    assert.equal(await publishPullRequest(f.args), 'skipped');
    assert.equal(f.calls.length, 0);
    assert.equal(f.fetched.length, 0);
});

test('PR polling skips forks, drafts, closed or outdated PRs, skip messages and failures', async () => {
    for (const option of ['fork', 'draft', 'closed', 'stale', 'movesOn', 'skip', 'failed']) {
        const f = fixture({[option]: true});
        assert.equal(await publishPullRequest(f.args), 'skipped', option);
        assert.equal(f.calls.length, 0, option);
    }
});

test('stop waiting when Buildkite never reports the current commit', async () => {
    const f = fixture({noStatus: true});
    await assert.rejects(publishPullRequest(f.args), /Timed out/);
    assert.equal(f.calls.length, 0);
});

test('replace only the author-posted bridge after the bot publishes successfully', async () => {
    const marker = '<!-- trixicuda-gpu-rhs-benchmarks -->';
    const bridge = `${marker}\nold\n_Posted from the completed Buildkite report as a pre-merge bridge; old_`;
    const comments = [
        {id: 1, user: {login: 'author'}, body: bridge},
        {id: 2, user: {login: 'someone-else'}, body: bridge},
        {id: 3, user: {login: 'author'}, body: `${marker}\nA human comment`},
    ];
    const f = fixture({comments});
    await publish(f.args);
    assert.deepEqual(f.calls.map((c) => c.operation), ['create', 'delete']);
    assert.equal(f.calls[1].comment_id, 1);
    const failed = fixture({comments});
    failed.args.github.rest.issues.createComment = async () => {throw new Error('No write permission');};
    await assert.rejects(publish(failed.args), /No write permission/);
    assert.equal(failed.calls.length, 0);
});
