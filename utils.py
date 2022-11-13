"""Helper functions."""
import gzip
import json
from pathlib import Path
import zipfile

import colorcet as cc
import pandas as pd
from scipy.stats import zscore

import json
import matplotlib.pyplot as plt


# Load configuration file so it's accessible from utils
with open("./config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def parse_text_id(text_id):
    """Convert filename to list of identifiers.
    in: corp-rdreams_auth-33_entry-2.txt
    out: [rdreams, 33, 2]
    """
    text_id = text_id.split(".")[0] # remove extension
    ids = text_id.split("_")
    corpus_id, author_id, entry_id = [ x.split("-")[1] for x in ids ]
    return [corpus_id, author_id, entry_id]

def load_zipped_corpus_info(corpus_id: str):
    """
    Return all non-data components of the standardized dataset (output from source2txt.py).
    This includes the attributes dataframe, metadata json, and list of filenames (i.e., entry IDs).
    """
    derivatives_directory = Path(config["derivatives_directory"])
    zip_path = derivatives_directory / "zips" / f"{corpus_id}.zip"
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        entry_ids = [ n.split(".txt")[0] for n in zf.namelist() if n.endswith(".txt") ]
        metadata = json.loads(zf.read(f"{corpus_id}.json"))
        with zf.open(f"{corpus_id}.tsv", mode="r") as f:
            attributes = pd.read_csv(f, sep="\t", index_col="text_id")
    assert len(entry_ids) == len(attributes)
    return entry_ids, metadata, attributes


def load_corpus_types():
    return { c: config["corpora"][c]["CorpusType"] for c in config["corpora"] }

def load_corpus_labels():
    return { c: config["corpora"][c]["LongName"] for c in config["corpora"] }

def load_types_palette():
    unique_corpus_types = sorted(set(load_corpus_types().values()))
    return { t: cc.cm.glasbey_dark(i) for i, t in enumerate(unique_corpus_types) }

def load_corpus_palette():
    return { c: cc.cm.glasbey_dark(i) for i, c in enumerate(config["corpora"]) }

def load_sourcedata(corpus_id, return_dataframe=True):
    """
    Return the original (source) data as a pandas dataframe,
    and the custom metadata dictionary (from config.json file).

    Option to return only metadata, since dataframes are sometimes large and not needed.
    """
    source_directory = Path(config["source_directory"])
    metadata = config["corpora"][corpus_id]
    source_name = metadata["SourceName"]
    import_path = source_directory / source_name

    if not return_dataframe:
        return metadata

    if corpus_id == "dreamviews":
        corpus = pd.read_csv(import_path, sep="\t")
    elif corpus_id == "dreamviewsann":
        corp1_path = import_path / "posts-annotations.csv"
        corp1 = pd.read_csv(corp1_path)
        corp1 = corp1.dropna(subset="text")
        corp1 = corp1.query("rater_id == 'hf24'") # or hacaldas
        corp1 = corp1.query("label != 'nondream'")
        corp1 = corp1.groupby(["report_id", "label"])["text"].sum()
        corp2_data = []
        for label in ["lucid", "nonlucid", "dream"]:
            corp2_paths = (import_path / f"mannheim_reports-{label}").glob("*.txt")
            for p in corp2_paths:
                report_id = p.stem
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
                if len(txt) > 30:
                    corp2_data.append({
                        "report_id": report_id,
                        "label": label,
                        "text": txt,
                    })
        corp2 = pd.DataFrame.from_records(corp2_data)
        corp2 = corp2.set_index(["report_id", "label"]).squeeze("columns")
        corpus = pd.concat([corp1, corp2], axis=0)
        corpus = corpus.reset_index()
    elif corpus_id == "thinkaloud":
        corpus = pd.read_excel(import_path)
        # Minor corrections that aren't being used now, so pointless.
        corpus = corpus.rename(columns={"Past ": "Past", "Past _NextThought": "Past_NextThought"})
        corpus["Random"] = corpus["Random"].replace({
                9: 0, # assuming bc buttons are close
            })
        corpus["ThoughtText"] = corpus["ThoughtText"].astype(str)
        corpus["ThoughtText"] = corpus["ThoughtText"].str.strip()
        # corpus = corpus.query("Taskrelated == 0")
        corpus = corpus.groupby(["ID", "TrialNumber"])["ThoughtText"].apply(". ".join)
        # corpus = corpus.groupby("ID")["ThoughtText"].apply(". ".join)
        corpus = corpus.reset_index(drop=False)
    elif corpus_id == "thoughtpings":
        corpus = pd.read_csv(import_path)
    elif corpus_id == "hippocorpus":
        with zipfile.ZipFile(import_path, mode="r") as zf:
            with zf.open("hcV3-stories.csv", mode="r") as f:
                corpus = pd.read_csv(f)
    else:
        raise ValueError(f"Unprepared for {corpus_id} corpus.")
    return corpus, metadata
