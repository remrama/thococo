"""Plot shuffled gutenberg coherence scores.
"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd

import utils


random.seed(0)
utils.load_matplotlib_settings()


# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / "corp-gutenberg_scores.tsv"
export_path_plot = deriv_dir / "corp-gutenberg_scores.png"

# Load data.
df = pd.read_csv(import_path, sep="\t")
_, _, attr = utils.load_zipped_corpus_info("gutenberg")
df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")

book_id_column = "title" # will be used for legend

df["title"] = df["title"].replace({
    "Twenty Thousand Leagues under the Sea": "20,000 Leagues",
    "The Merry Adventures of Robin Hood": "Robin Hood",
})

df = df.query("id.ne('PG141')")
df = df.query("id.ne('PG2641')")
df = df.query("id.ne('PG5230')")
df = df.query("id.ne('PG42324')")

df = df.sort_values("IncoherenceMean", ascending=False)

shuffle_rates = df["shuffle_rate"].sort_values().unique()

n_books = df["id"].nunique()
all_markers = list(plt.matplotlib.lines.Line2D.markers)[2:]
all_markers = all_markers[:n_books]
random.shuffle(all_markers)
markers = { book: all_markers[i] for i, book in enumerate(df[book_id_column].unique()) }



# Plot.

fig, ax = plt.subplots(figsize=(2.7, 2))

for book_id, book_df in df.groupby(book_id_column):
    book_df = book_df.sort_values("shuffle_rate")
    m = markers[book_id]
    xvals = book_df["shuffle_rate"].to_numpy()
    yvals = book_df["IncoherenceMean"].to_numpy()
    # fmt = f"-{m}"
    ax.plot(xvals, yvals, marker=m, label=book_id,
        mew=0.5, lw=0.5, ms=4, color="black", mec="black", mfc="white")

ax.set_ylabel("Semantic incoherence")
ax.set_xlabel("Proportion of sentences shuffled")
ax.margins(x=0.1)
ax.set_ylim(0.43, 0.61)
# ax.xaxis.set(major_locator=plt.FixedLocator(shuffle_rates))
ax.xaxis.set(major_locator=plt.MultipleLocator(0.5),
             minor_locator=plt.MultipleLocator(0.1))
ax.yaxis.set(major_locator=plt.MultipleLocator(0.05),
             minor_locator=plt.MultipleLocator(0.01))

handles = [ plt.matplotlib.lines.Line2D([], [],
        marker=m, label=b, markersize=4,
        color="white", mec="black", mew=0.5, linestyle="None")
    for b, m in markers.items() ]
ax.legend(handles=handles,
    # title="Book title",
    loc="upper left", bbox_to_anchor=(1, 1),
    borderaxespad=-0.5, frameon=False,
    labelspacing=0.2,  # rowspacing, vertical space between the legend entries
    handletextpad=-0.2,  # space between legend marker and label
    fontsize=6,
)


# Export.
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()
