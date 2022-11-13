# Visualize relationship between self-reported mind-wandering and semantic coherence.
from pathlib import Path
import pandas as pd
import utils

import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns

sns.set_style("white")
plt.ion()

corpus_id = "thoughtpings"
attribute = "wandering"
scorer = "SequentialCoherenceMean"

# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / f"corp-{corpus_id}_scores.tsv"
export_path = deriv_dir / f"corp-{corpus_id}_attr-{attribute}_corr.png"

# Load data.
df = pd.read_csv(import_path, sep="\t")

_, _, attr = utils.load_zipped_corpus_info(corpus_id)
attr = attr.dropna(subset=attribute)
# df = df.set_index(ids).join(attr.set_index(ids)["wandering"], how="inner")
df = df.merge(attr, on=["corpus_id", "author_id", "entry_id"], how="inner")

# Without subject averaging.
# method="spearman", 
corr = pg.rm_corr(data=df, x=attribute, y=scorer, subject="author_id")
r, p = corr.loc["rm_corr", ["r", "pval"]]
g = pg.plot_rm_corr(data=df, x=attribute, y=scorer, subject="author_id",
    kwargs_facetgrid=dict(height=3, aspect=1, palette="cubehelix"))

txt = f"r = {r:.2f}\np = {p:.3f}"
g.ax.text(.99, .99, txt, transform=g.ax.transAxes, ha="right", va="top")
g.ax.set_xticks(range(1, 7))
g.ax.set_xlabel("Self-reported Mind-wandering")
g.ax.set_ylabel("Semantic coherence")

# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()