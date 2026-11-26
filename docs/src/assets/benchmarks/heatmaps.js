// Draws the sweep heatmaps into #sweep-heatmaps. The data comes from sweep-data.js, which
// docs/make.jl writes next to this file at build time (null when no sweep has run yet).
(function () {
  "use strict";
  const dataUrl = new URL("sweep-data.js", document.currentScript.src).href;

  const SWEEPS = [
    { key: "level", title: "Level sweep", column: (p) => p.level, axis: "L", fixed: (p) => `p = ${p.polydeg}` },
    { key: "degree", title: "Degree sweep", column: (p) => p.polydeg, axis: "p", fixed: (p) => `L = ${p.level}` },
  ];
  const VERDICTS = ["faster", "slower", "no change", "inconclusive"];

  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    node.append(...children);
    return node;
  };
  const ndims = (name) => Number(name.match(/_(\d)d$/)[1]);
  const shortName = (name) => name.replace(/_\dd$/, "");
  const sha = (commit) => el("code", {}, commit.slice(0, 7));

  // Blue for faster, red for slower; the colour deepens up to a 2x change.
  function fill(point) {
    if (point.verdict !== "faster" && point.verdict !== "slower") return "";
    const strength = Math.min(1, Math.abs(Math.log2(point.speedup)));
    const rgb = point.verdict === "faster" ? "37, 99, 235" : "220, 38, 38";
    return `background-color: rgba(${rgb}, ${(0.2 + 0.7 * strength).toFixed(2)})`;
  }

  function cell(point) {
    if (!point) return el("td", { class: "outside", title: "Not in the sweep grid" });
    const where = `${point.case}, p = ${point.polydeg}, L = ${point.level}`;
    if (point.error) return el("td", { class: "failed", title: `${where}\n${point.error}` }, "×");
    const [low, high] = point.interval;
    const title = `${where}\n${point.dofs_per_field.toLocaleString("en-US")} DOFs per field\n` +
      `Speedup ${point.speedup.toFixed(3)}x, 95% interval ${low.toFixed(3)}–${high.toFixed(3)}: ${point.verdict}`;
    const strong = point.style !== "" && Math.abs(Math.log2(point.speedup)) > 0.6;
    const classes = [point.verdict.replace(" ", "-"), strong ? "strong" : ""].join(" ").trim();
    return el("td", { class: classes, style: point.style, title }, point.speedup.toFixed(2));
  }

  function panel(sweep, d, points) {
    const columns = [...new Set(points.map(sweep.column))].sort((a, b) => a - b);
    const cases = [...new Set(points.map((p) => p.case))];
    const head = el("tr", {}, el("th", {}, sweep.axis), ...columns.map((c) => el("th", {}, String(c))));
    const rows = cases.map((name) => {
      const byColumn = new Map(points.filter((p) => p.case === name).map((p) => [sweep.column(p), p]));
      return el("tr", {}, el("th", {}, shortName(name)), ...columns.map((c) => cell(byColumn.get(c))));
    });
    const caption = el("caption", {}, `${d}D, ${sweep.fixed(points[0])}`);
    return el("div", { class: "sweep-panel" },
      el("table", { class: "sweep-heatmap" }, caption, el("thead", {}, head), el("tbody", {}, ...rows)));
  }

  function summary(data) {
    const { baseline, target, points } = data;
    const counts = VERDICTS.map((v) => `${points.filter((p) => p.verdict === v).length} ${v}`);
    const failed = points.filter((p) => p.error).length;
    if (failed) counts.push(`${failed} failed`);
    const run = /^https:\/\//.test(target.build_url) ? el("a", { href: target.build_url }, "measurement run") : "";
    return el("p", {},
      el("strong", {}, "Baseline "), sha(baseline.commit), el("strong", {}, " → Target "), sha(target.commit),
      `, measured on ${target.gpu} on ${target.measured_at.slice(0, 10)} (`, run, `). ` +
      `${points.length} points: ${counts.join(", ")}.`);
  }

  function legend(band) {
    const pct = `±${Math.round(100 * band)}%`;
    const item = (cls, style, text, label) =>
      el("span", { class: "sweep-legend-item" }, el("span", { class: `sweep-swatch ${cls}`, style }, text), label);
    return el("p", { class: "sweep-legend" },
      item("faster", fill({ verdict: "faster", speedup: 1.25 }), "", `faster (whole interval above ${pct})`),
      item("slower", fill({ verdict: "slower", speedup: 0.8 }), "", `slower (whole interval below ${pct})`),
      item("no-change", "", "", `no change (interval inside ${pct})`),
      item("inconclusive", "", "", "inconclusive"),
      item("failed", "", "×", "failed"),
      item("outside", "", "", "not in the grid"));
  }

  function render(root, data) {
    if (!data) {
      root.append(el("p", {}, "No sweep results yet. Push a commit whose title contains ",
        el("code", {}, "[benchmark sweep]"), " to measure one."));
      return;
    }
    data.points.forEach((p) => { p.style = p.error ? "" : fill(p); });
    root.append(summary(data), legend(data.noise_band));
    for (const sweep of SWEEPS) {
      root.append(el("h3", {}, sweep.title));
      for (const d of [1, 2, 3]) {
        const points = data.points.filter((p) => p.sweeps.includes(sweep.key) && ndims(p.case) === d);
        if (points.length) root.append(panel(sweep, d, points));
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("sweep-heatmaps");
    if (!root) return;
    const script = el("script", { src: dataUrl });
    script.onload = () => render(root, window.SWEEP_COMPARISON);
    script.onerror = () => render(root, null);
    document.head.append(script);
  });
})();
