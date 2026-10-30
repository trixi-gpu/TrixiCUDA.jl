import assert from 'node:assert/strict';
import test from 'node:test';
import {formatComment, parseReport, publish} from './benchmark-comment.mjs';

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
    const github = {
        rest: {
            pulls: {get: async () => ({data: {
                state: options.closed ? 'closed' : 'open', draft: options.draft ?? false,
                head: {sha: options.stale || (options.movesOn && reads++ > 0) ? baseline : target},
            }})},
            issues: {
                listComments: 'listComments',
                createComment: async (args) => calls.push({operation: 'create', ...args}),
                updateComment: async (args) => calls.push({operation: 'update', ...args}),
            },
        },
        paginate: async () => options.existing ? [{id: 123, user: {login: 'github-actions[bot]'},
            body: '<!-- trixicuda-gpu-rhs-benchmarks -->\nold'}] : [],
    };
    const fetched = [];
    const fetchImpl = async (url) => {
        fetched.push(url);
        let data;
        if (url.endsWith('.json')) data = {commit_id: options.wrongCommit ? baseline : target,
            pull_request: options.main ? null : {id: '227'}};
        else if (url.endsWith('/steps')) data = [{key: 'benchmarks', outcome: options.failed ? 'soft_failed' : 'passed'}];
        else if (url.endsWith('/annotations')) data = {records: options.missing ? [] : [{context: 'gpu-rhs-benchmarks', body: {html}}]};
        else throw new Error(`Unexpected URL: ${url}`);
        return {ok: true, json: async () => data};
    };
    const context = {repo: {owner: 'trixi-gpu', repo: 'TrixiCUDA.jl'}, payload: {sha: target, target_url: options.url ?? buildUrl}};
    return {args: {github, context, fetchImpl, log: () => {}}, calls, fetched};
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

test('no comment for main, drafts, merged PRs, stale heads or failed/missing benchmarks', async () => {
    for (const option of ['main', 'draft', 'closed', 'stale', 'movesOn', 'failed', 'missing']) {
        const f = fixture({[option]: true});
        await publish(f.args);
        assert.equal(f.calls.length, 0, option);
        if (option === 'main') assert.equal(f.fetched.length, 1);
    }
});

test('reject unrelated Buildkite URLs and commit mismatches', async () => {
    const wrongURL = fixture({url: 'https://example.com/julialang/trixicuda-dot-jl/builds/313'});
    await assert.rejects(publish(wrongURL.args));
    assert.equal(wrongURL.fetched.length, 0);
    const wrongCommit = fixture({wrongCommit: true});
    await assert.rejects(publish(wrongCommit.args));
    assert.equal(wrongCommit.calls.length, 0);
});
