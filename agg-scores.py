"""Get some summary stats for each corpus.

Table 1:
    - n_reports per author (per corpus)
    - mean, std, etc. character length per author (per corpus)

Table 2:
    - n_reports per corpus
    - n_authors per corpus
    - total character length per corpus
    - mean, std, etc. n_reports per author
    - mean, std, etc. character length per corpus
"""
from pathlib import Path
import pandas as pd
import utils


# Declare filepaths.
deriv_dir = Path(utils.config["derivatives_directory"])
import_paths = deriv_dir.glob("corp-*_scor-*.tsv")

export_path_auth = deriv_dir / "agg-auth_scores.tsv"
export_path_corp = deriv_dir / "agg-corp_scores.tsv"

# scores = utils.config["scores"]
# scores_map = {}
# for scorefam, score_list in scores.items():
#     for s in score_list:
#         assert s not in scores_map, f"{s} can't be reused"
#         scores_map[s] = scorefam

# Load data.
# df = pd.concat([ pd.read_csv(p).assign(scores_id=p.stem.split("_scor-")[1]) for p in import_paths ])
df = pd.concat(
    [ pd.read_csv(p, sep="\t").melt(id_vars=["corpus_id", "author_id", "entry_id"], var_name="scorer", value_name="score")
        for p in import_paths ],
    axis=0)

# df["scorer_family"] = df["scorer"].map(scores_map)
# df = df.set_index("scorer_family", append=True)


########### Aggregate by author


# auth_desc = df.groupby(["corpus_id", "author_id"]
#     )["character_length"].agg(["count", "mean", "std", "median", "min", "max"])
auth = df.groupby(["corpus_id", "author_id", "scorer"])["score"].agg(["count", "mean", "std", "sem", "median", "min", "max"])

# # auth = df.groupby(["corpus_id", "author_id", "scorer"])["score"].mean()
# auth = auth.unstack("scorer").rename_axis(["agg_author", "scorer"], axis=1
#     ).swaplevel(axis="columns"

# auth.columns = auth.columns.swaplevel().map("-".join)
# auth = auth.rename(columns={"Ncharacters-count": "n_entries"})
# # auth = auth.filter(items=[ c for c in auth if not c.endswith("_count") ], axis=1)
# auth = auth.drop(columns=[ c for c in auth if c.endswith("-count") ])


# auth.groupby(["corpus_id", "scorer"])["mean"].agg(["count", "mean", "std", "min"])
corp = auth["mean"].groupby(["corpus_id", "scorer"]
    ).agg(["count", "mean", "std", "sem", "median", "min", "max"]
    ).rename(columns={"count": "n_authors"})

corp.insert(0, "n_entries", auth["count"].groupby(["corpus_id", "scorer"]).sum())


auth.to_csv(export_path_auth, index=True, sep="\t")
corp.to_csv(export_path_corp, index=True, sep="\t")
