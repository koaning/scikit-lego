import sys

import altair as alt
import polars as pl

KMEANS = {
    "Lloyd KMeans (random)": "Lloyd KMeans, random init",
    "Lloyd KMeans (k-means++)": "Lloyd KMeans, k-means++ init",
    "MiniBatchKMeans (B=4096)": "MiniBatchKMeans",
    "EMeans sklego (random, revival)": "EMeans (this PR, NumPy)",
    "emeans Rust crate (random, revival)": "e-means (Rust crate)",
    "emeans Rust crate (random, no revival)": "e-means (Rust crate), revival off",
}
GMM = {
    "EM GaussianMixture (init=kmeans)": "EM, sklearn default (k-means init)",
    "EM GaussianMixture (init=random_from_data)": "EM, random-row init",
    "EM GaussianMixture (init=e-means)": "EM, e-means init",
    "e-GMM (init=random rows, law)": "e-GMM, random-row init, k-means laws (alpha=1)",
    "e-GMM (init=random rows, B=16384, a=0.1)": "e-GMM, random-row init, B=16384 alpha=0.1",
    "e-GMM (init=random rows, B=16384, a=0.3)": "e-GMM, random-row init, B=16384 alpha=0.3",
    "e-GMM (init=e-means, B=16384, a=0.3)": "e-GMM, e-means init, B=16384 alpha=0.3",
}


def panel(df, title, y, ytitle, ref_method, ref_iters, unit, higher_better, dflt=None):
    ref = df.filter(pl.col("method") == ref_method, pl.col("iters") == ref_iters)[y][0]
    cut = ref - CUT * abs(ref) if higher_better else ref * (1 + CUT)
    hidden = df.filter(pl.col(y) < cut if higher_better else pl.col(y) > cut).height
    df = df.filter(pl.col(y) >= cut if higher_better else pl.col(y) <= cut)
    df = df.with_columns(
        label=pl.when(pl.col("method").str.contains("^(e-|EMeans)"))
        .then(pl.col("iters").cast(pl.Utf8) + " ep")
        .otherwise(pl.col("iters").cast(pl.Utf8) + " it")
    )
    base = alt.Chart(df).encode(
        x=alt.X("seconds:Q", scale=alt.Scale(type="log"), title="training time incl. init (seconds, log scale)"),
        y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), title=ytitle),
        color=alt.Color("method:N", title=None, legend=alt.Legend(orient="bottom", columns=2, labelLimit=400)),
    )
    lines = base.mark_line(point=alt.OverlayMarkDef(size=50))
    text = base.mark_text(dx=7, dy=-7, fontSize=10, align="left").encode(text="label:N")
    rule = alt.Chart(pl.DataFrame({y: [ref]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y=f"{y}:Q")
    rule_txt = (
        alt.Chart(pl.DataFrame({y: [ref], "t": [f"reference: {ref_method.split(' (')[0]} {ref_iters} it"]}))
        .mark_text(align="left", dy=-6, x=5, color="gray")
        .encode(y=f"{y}:Q", text="t:N")
    )
    layers = rule + rule_txt + lines + text
    if dflt is not None and dflt.height:
        d = dflt.rename({"test": y}).with_columns(
            label=pl.lit("sklearn defaults (") + pl.col("iters").cast(pl.Utf8) + " it)"
        )
        star = alt.Chart(d).encode(x="seconds:Q", y=f"{y}:Q")
        layers = (
            layers
            + star.mark_point(shape="diamond", size=160, filled=True, color="black")
            + star.mark_text(dx=8, dy=10, align="left", fontWeight="bold").encode(text="label:N")
        )
    return (layers).properties(
        width=460,
        height=340,
        title=alt.TitleParams(
            title,
            subtitle=[
                "each point is one fit; label = iterations (it) or epochs (ep)",
                f"{hidden} fits more than {CUT:.1%} worse than the dashed reference are not shown",
            ],
        ),
    )


kind = sys.argv[1]
CUT = 0.015 if kind == "kmeans" else 0.05
import os

DFLT = (
    pl.read_csv("results/sklearn_defaults.csv").filter(pl.col("kind") == ("kmeans" if kind == "kmeans" else "gmm"))
    if os.path.exists("results/sklearn_defaults.csv")
    else pl.DataFrame({"dataset": [], "seconds": [], "test": [], "iters": []})
)
if kind == "kmeans":
    df = pl.read_csv("results/kmeans_glove.csv").filter(pl.col("method").is_in(list(KMEANS)))
    df = df.with_columns(pl.col("method").replace(KMEANS))
    charts = [
        panel(
            df.filter(pl.col("dataset") == d),
            d.replace("400k x 100", "320k train / 80k test"),
            "test_mse",
            "test MSE: mean squared distance to nearest centroid (lower is better)",
            "Lloyd KMeans, k-means++ init",
            30,
            " it",
            False,
            DFLT.filter(pl.col("dataset") == d),
        )
        for d in sorted(df["dataset"].unique(), key=lambda s: "1024" not in s)
    ]
else:
    df = (
        pl.concat(
            [
                pl.read_csv(f).select("dataset", "method", "iters", "seconds", "test_ll", "train_ll")
                for f in sys.argv[2:-1]
            ]
        )
        .group_by("dataset", "method", "iters")
        .agg(pl.col("seconds", "test_ll", "train_ll").median())
    )
    df = (
        df.filter(pl.col("method").is_in(list(GMM))).with_columns(pl.col("method").replace(GMM)).sort("method", "iters")
    )
    charts = [
        panel(
            df.filter(pl.col("dataset") == d),
            d,
            "test_ll",
            "held-out log-likelihood per row (higher is better)",
            "EM, sklearn default (k-means init)",
            int(df.filter(pl.col("dataset") == d, pl.col("method").str.contains("default"))["iters"].max()),
            " it",
            True,
            DFLT.filter(pl.col("dataset") == d),
        )
        for d in sorted(df["dataset"].unique())
    ]
alt.hconcat(*charts).resolve_scale(color="shared").save(sys.argv[-1], scale_factor=1.6)
