# ruff: noqa: T201
"""Charts and tables from the CSVs written by `run.py`.

```
python benchmarks/emeans/report.py            # reads results/{kmeans,gmm}.csv
python benchmarks/emeans/report.py --quick    # reads results/{kmeans,gmm}_quick.csv
```

Writes `kmeans.png`, `gmm.png` and `tables.md` next to the CSVs. Every number is a mean over seeds.
"""

import argparse

import altair as alt
import polars as pl
from run import CONVERGED, GMM_METHODS, KMEANS_METHODS, RESULTS

# Categorical palette in fixed order (colour follows the algorithm, in the order the registries list them).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, SURFACE, BAND = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3"
DATA_ORDER = ["embedding-like", "covtype", "blobs"]


def algorithm_colors(methods):
    algorithms = list(dict.fromkeys(m.algorithm for m in methods if not m.reference))
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(algorithms)}


def mean_curve(df, keys, quality):
    """One row per (keys, method, budget): mean time and quality over seeds."""
    return df.group_by([*keys, "method", "algorithm", "init", "unit", "reference", "budget"], maintain_order=True).agg(
        seconds=pl.col("seconds").mean(),
        quality=pl.col(quality).mean(),
        passes=pl.col("passes").mean(),
        ari=pl.col("ari").cast(pl.Float64, strict=False).mean(),
        seeds=pl.len(),
    )


def markdown(df):
    header = "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"] + ["--:"] * (df.width - 1)) + "|\n"
    return header + "".join(
        "| " + " | ".join("–" if v is None else str(v) for v in row) + " |\n" for row in df.iter_rows()
    )


def summary_tables(curve, group, columns, higher_is_better, tolerance, gap):
    """Two tables, methods as rows and `columns` as columns.

    `reach`: seconds until the fit first gets within `tolerance` of the reference. `best`: the best quality reached
    at any budget, relative to the reference.
    """
    reach, best = [], []
    for column in columns:
        g = curve.filter(pl.col(group) == column)
        reference = g.filter(pl.col("reference")).row(0, named=True)
        within = (
            pl.col("quality") >= reference["quality"] - tolerance
            if higher_is_better
            else pl.col("quality") <= reference["quality"] * (1 + tolerance / 100)
        )
        for method in g.filter(~pl.col("reference"))["method"].unique(maintain_order=True):
            m = g.filter(pl.col("method") == method)
            hit = m.filter(within)
            reach.append(
                {"method": method, group: column, "v": f"{hit['seconds'].min():.2f}s" if hit.height else "never"}
            )
            top = m.sort("quality", descending=higher_is_better).row(0, named=True)
            best.append({"method": method, group: column, "v": gap(top["quality"], reference["quality"])})
        reach.append(
            {
                "method": f"{reference['method']} (reference)",
                group: column,
                "v": f"{reference['seconds']:.2f}s, {reference['passes']:.0f} it",
            }
        )

    def pivot(rows):
        return pl.DataFrame(rows).pivot(on=group, index="method", values="v").select(["method", *columns])

    return pivot(reach), pivot(best)


def reference_layers(text, tolerance):
    band = (
        alt.Chart(pl.DataFrame({"lo": [-tolerance], "hi": [tolerance]}))
        .mark_rect(color=BAND, opacity=0.45)
        .encode(y="lo:Q", y2="hi:Q")
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[3, 3], color=INK, strokeWidth=1.2).encode(y="y:Q")
    )
    label = (
        alt.Chart(pl.DataFrame({"y": [0.0], "t": [text]}))
        .mark_text(align="left", x=6, dy=-7, fontSize=10, color=INK)
        .encode(y="y:Q", text="t:N")
    )
    return band + rule + label


def panel(points, reference, title, y_title, y_scale, colors, tolerance):
    methods = points["method"].unique(maintain_order=True).to_list()
    by_method = points.unique("method", maintain_order=True)
    color = alt.Color(
        "method:N",
        scale=alt.Scale(domain=methods, range=[colors[a] for a in by_method["algorithm"]]),
        legend=alt.Legend(title=None, orient="bottom", columns=4, symbolType="stroke", symbolSize=400),
    )
    dash = alt.StrokeDash(
        "method:N",
        scale=alt.Scale(
            domain=methods,
            range=[[1, 0] if i in ("k-means++", "e-means", "k-means") else [5, 3] for i in by_method["init"]],
        ),
        legend=None,
    )
    base = alt.Chart(points).encode(
        x=alt.X("seconds:Q", scale=alt.Scale(type="log"), title="training time incl. init (s, log)"),
        y=alt.Y("gap:Q", scale=y_scale, title=y_title),
        color=color,
    )
    marks = base.mark_line(strokeWidth=2).encode(strokeDash=dash, detail="method:N") + base.mark_point(
        filled=True, size=45, stroke=SURFACE, strokeWidth=1.5, opacity=1
    )
    text = f"{reference['method']}: {reference['passes']:.0f} it, {reference['seconds']:.1f}s"
    return (reference_layers(text, tolerance) + marks).properties(width=330, height=230, title=title)


def save(chart, path, title, subtitle):
    (
        chart.properties(title=alt.Title(title, subtitle=subtitle))
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#ecebe8", domainColor="#c9c8c3", labelColor=MUTED, titleColor=MUTED, titleFontWeight="normal"
        )
        .configure_title(color=INK, subtitleColor=MUTED, anchor="start")
        .configure_legend(labelColor=INK, symbolStrokeWidth=2.5, labelLimit=300)
        .save(path, scale_factor=2)
    )


def kmeans_report(df, png, tolerance):
    curve = mean_curve(df, ["data"], "test_mse")
    data = [d for d in DATA_ORDER if d in curve["data"].to_list()]
    colors = algorithm_colors(KMEANS_METHODS)
    panels = []
    for d in data:
        g = curve.filter(pl.col("data") == d)
        reference = g.filter(pl.col("reference")).row(0, named=True)
        points = g.filter(~pl.col("reference")).with_columns(gap=(pl.col("quality") / reference["quality"] - 1) * 100)
        k = df.filter(pl.col("data") == d)["k"][0]
        panels.append(
            panel(
                points,
                reference,
                f"{d} (k={k})",
                "test MSE above the reference (%, symlog)",
                alt.Scale(type="symlog", constant=0.1),
                colors,
                tolerance,
            )
        )
    save(
        alt.hconcat(*panels).resolve_scale(y="independent"),
        png,
        "k-means: held-out MSE against training time",
        [
            "Lines: each method stopped after a fixed budget of iterations (it) or epochs (ep); mean over seeds.",
            f"Dotted line: KMeans() with sklearn's defaults, which iterates until it converges. Gray band: within ±{tolerance}% of it. Lower and further left is better.",
        ],
    )
    reach, best = summary_tables(curve, "data", data, False, tolerance, lambda q, r: f"{(q / r - 1) * 100:+.2f}%")
    text = f"### Time to get within ±{tolerance}% of the test MSE of `KMeans()`\n\n{markdown(reach)}\n"
    text += f"### Best test MSE at any budget, relative to `KMeans()` (lower is better)\n\n{markdown(best)}\n"
    if "blobs" in data:
        ari = (
            curve.filter(pl.col("data") == "blobs").sort("quality").unique("method", keep="first", maintain_order=True)
        )
        text += "### Cluster recovery on blobs: ARI of each method's best fit against the true labels\n\n"
        text += markdown(ari.select("method", pl.col("ari").round(3).alias("ARI"))) + "\n"
    return text


def gmm_report(df, png, tolerance):
    mixtures = df.filter(pl.col("covariance").is_in(["diag", "spherical"]))
    curve = mean_curve(
        mixtures.with_columns(column=pl.col("data") + " / " + pl.col("covariance")),
        ["column", "data", "covariance"],
        "test_ll",
    )
    data = [d for d in DATA_ORDER if d in curve["data"].to_list()]
    columns = [f"{d} / {c}" for c in ["diag", "spherical"] for d in data]
    colors = algorithm_colors(GMM_METHODS)
    rows = []
    for covariance in ["diag", "spherical"]:
        panels = []
        for d in data:
            g = curve.filter(pl.col("column") == f"{d} / {covariance}")
            reference = g.filter(pl.col("reference")).row(0, named=True)
            points = (
                g.filter(~pl.col("reference"))
                .with_columns(gap=pl.col("quality") - reference["quality"])
                .filter(pl.col("gap") >= -0.5)
            )
            lo, hi = min(points["gap"].min(), -3 * tolerance), max(points["gap"].max(), 3 * tolerance)
            pad = 0.1 * (hi - lo)
            panels.append(
                panel(
                    points,
                    reference,
                    f"{d} / {covariance}",
                    "test log-lik. vs reference (nats/row)",
                    alt.Scale(domain=[lo - pad, hi + pad], nice=False),
                    colors,
                    tolerance,
                )
            )
        rows.append(alt.hconcat(*panels).resolve_scale(y="independent"))
    save(
        alt.vconcat(*rows),
        png,
        "Gaussian mixtures: held-out log-likelihood against training time",
        [
            "Lines: each method stopped after a fixed budget of EM iterations (it, tol=0) or epochs (ep); mean over seeds.",
            f"Dotted line: GaussianMixture() with sklearn's defaults (stops when the gain per iteration drops below 1e-3). Gray band: within ±{tolerance} nats/row. Higher is better.",
        ],
    )

    reach, best = summary_tables(curve, "column", columns, True, tolerance, lambda q, r: f"{q - r:+.3f}")
    text = f"### Time to get within {tolerance} nats/row of the test log-likelihood of `GaussianMixture()`\n\n{markdown(reach)}\n"
    text += f"### Best test log-likelihood at any budget, nats/row relative to `GaussianMixture()` (higher is better)\n\n{markdown(best)}\n"

    full = df.filter(pl.col("covariance") == "full")
    if full.height:
        diag = curve.filter(pl.col("covariance") == "diag", pl.col("reference")).select("data", diag=pl.col("quality"))
        full = full.group_by("data").agg(pl.col("test_ll").mean(), pl.col("seconds").mean()).join(diag, on="data")
        text += '### Full covariance, for comparison: `GaussianMixture(covariance_type="full")` defaults\n\n'
        text += "EGaussianMixture does not support full covariances. This is what one would add over the diagonal default.\n\n"
        text += (
            markdown(
                full.select(
                    "data",
                    (pl.col("test_ll") - pl.col("diag")).round(2).alias("nats/row over diag"),
                    pl.col("seconds").round(1).alias("seconds"),
                )
            )
            + "\n"
        )

    realism = df.filter(pl.col("realism").is_not_null())
    if realism.height:
        label = (
            pl.when(pl.col("method") == "real rows (floor)")
            .then(pl.lit("real rows (floor)"))
            .otherwise(
                pl.col("method")
                + " / "
                + pl.col("covariance")
                + pl.when(pl.col("budget") == CONVERGED).then(pl.lit("")).otherwise(", " + pl.col("budget") + " ep")
            )
        )
        table = (
            realism.with_columns(model=label)
            .group_by("model", "data", maintain_order=True)
            .agg(pl.col("realism").mean().round(3))
        )
        table = table.pivot(on="data", index="model", values="realism").select(
            ["model", *[d for d in DATA_ORDER if d in table["data"].to_list()]]
        )
        text += "### Sample realism: AUC of a classifier telling 20k samples from 20k real held-out rows (0.5 = indistinguishable)\n\n"
        text += (
            "covtype's columns are integer-valued, so Gaussian samples are always easy to spot there.\n\n"
            + markdown(table)
            + "\n"
        )

    if "blobs" in data:
        ari = (
            curve.filter(pl.col("data") == "blobs")
            .sort("quality", descending=True)
            .unique(["covariance", "method"], keep="first", maintain_order=True)
        )
        text += "### Cluster recovery on blobs: ARI of each method's best fit against the true labels\n\n"
        text += (
            markdown(
                ari.sort("covariance", "method").select("covariance", "method", pl.col("ari").round(3).alias("ARI"))
            )
            + "\n"
        )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kmeans-tolerance", type=float, default=0.5, help="in %% of the reference test MSE")
    parser.add_argument("--gmm-tolerance", type=float, default=0.02, help="in nats/row of test log-likelihood")
    args = parser.parse_args()
    suffix = "_quick" if args.quick else ""
    parts = []
    for task, report, tolerance in [
        ("kmeans", kmeans_report, args.kmeans_tolerance),
        ("gmm", gmm_report, args.gmm_tolerance),
    ]:
        path = RESULTS / f"{task}{suffix}.csv"
        if path.exists():
            df = pl.read_csv(path, infer_schema_length=None)
            seeds = df["seed"].n_unique()
            parts.append(
                f"## {'k-means' if task == 'kmeans' else 'Gaussian mixtures'}\n\n"
                f"Machine: {df['machine'][0]}. Mean over {seeds} seed{'s' if seeds > 1 else ''}. "
                f"![{task}]({task}{suffix}.png)\n\n" + report(df, RESULTS / f"{task}{suffix}.png", tolerance)
            )
    (RESULTS / f"tables{suffix}.md").write_text(
        "# Benchmark results\n\nGenerated by `report.py`, do not edit.\n\n" + "\n".join(parts)
    )
    print(f"wrote {RESULTS}/tables{suffix}.md")


if __name__ == "__main__":
    main()
