"""Visualize the relationship between number of reports
and number of users across multiple corpora.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import utils


# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / "agg-corp_scores.tsv"
export_path = deriv_dir / "agg-corp_samplesize.png"

# Load data.
df = pd.read_csv(import_path, sep="\t")
df = df.sort_index(axis=0).sort_index(axis=1)


# Numbers will slightly vary depending on how many reports meet the requirements for each scorer analysis.
# Character count is the least restrictive and should include all reports.
df = df.query("scorer=='SequentialCoherenceN'")
df = df[["corpus_id", "n_entries", "n_authors"]]

corpora = [
    "thinkaloud",
    "thoughtpings",
    "dreamviews",
    "dreambank",
    "rdreams",
    "cosowell",
    "hippocorpus",
    "emails",
    "speeches",
]
# corpora = [
#     "dreambank",
#     "sddb",
#     "dreamviews",
#     "rdreams",
#     "dreamscloud",
# ]
df = df.query(f"corpus_id.isin({corpora})")

labels_map = utils.load_corpus_labels()
types_map = utils.load_corpus_types()
palette_map = utils.load_types_palette()
markers = list(plt.matplotlib.lines.Line2D.markers)[2:]
# palette_map = utils.load_corpus_palette()
# markers = ["+"] * 100
markers_map = { c: markers[i] for i, c in enumerate(corpora) }

df["label"] = df["corpus_id"].map(labels_map)
df["ctype"] = df["corpus_id"].map(types_map)
df["color"] = df["ctype"].map(palette_map)
# df["color"] = df["corpus_id"].map(palette_map)
df["marker"] = df["corpus_id"].map(markers_map)

df = df.sort_values(["ctype", "corpus_id"])
# df = df.set_index("corpus_id").reindex(corpora).reset_index()

# Declare plotting variables
dmlab.plotting.set_matplotlib_style("technical")
# figsize = (6, 6)
figsize = (4, 2.5)
low_ax_limit = 1
high_ax_limit = 1000000
xscale = "log"
yscale = "log"
xlabel = "Number of unique documents"
ylabel = "Number of unique authors"
legend_title = "Corpus"
scatter_kwargs = dict(s=80, zorder=2)
line_kwargs = dict(color="black", linewidth=1, zorder=1)

# Draw.
fig, ax = plt.subplots(figsize=figsize)
# ax.scatter("n_entries", "n_authors", color="color", marker="+", data=df, **scatter_kwargs)
for _, row in df.iterrows():
    ax.scatter(row["n_entries"], row["n_authors"],
        color=row["color"], marker=row["marker"], label=row["label"],
        **scatter_kwargs)
ax.plot([low_ax_limit, high_ax_limit], [low_ax_limit, high_ax_limit], "--", **line_kwargs)

# Aesthetics.
# ax.set_aspect(1)
ax.grid(axis="x")
ax.set(
    aspect=1,
    xlim=(low_ax_limit, high_ax_limit),
    ylim=(low_ax_limit, high_ax_limit),
    xscale=xscale,
    yscale=yscale,
    xlabel=xlabel,
    ylabel=ylabel,
)

# Legend
# handles = [ plt.matplotlib.patches.Patch(edgecolor="black",
#         linewidth=.5, facecolor=row["color"], label=row["label"])
#     for _, row in df.iterrows() ]
# legend = ax.legend(handles=handles, title=legend_title,
#     # bbox_to_anchor=(.02, 1), loc="upper left")
#     bbox_to_anchor=(1, 1), loc="upper left")
# legend._legend_box.align = "left"

# handles = [ plt.Line2D([0], [0], marker="o", color=c, label=l, markersize=5, linewidth=0)
#     for l, c in palette.items() ]
handles = [ plt.matplotlib.lines.Line2D([0], [0],
        marker=row["marker"], color=row["color"], label=row["label"],
        markersize=6, linewidth=0)
    for _, row in df.iterrows() ]
# legend = ax.legend(handles=handles, title=legend_title,
#     # bbox_to_anchor=(.02, 1), loc="upper left")
#     bbox_to_anchor=(1, 1), loc="upper left")
# legend._legend_box.align = "left"
legend = ax.legend(
    handles=handles,
    title=legend_title,
    bbox_to_anchor=(1, 1), loc="upper left",
    labelspacing=0.1,  # rowspacing, vertical space between the legend entries
    handletextpad=0.2,  # space between legend marker and label
)
legend._legend_box.align = "left"

# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()
