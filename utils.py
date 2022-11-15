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
    elif corpus_id == "thoughtpings":
        corpus = pd.read_csv(import_path)
    elif corpus_id == "hippocorpus":
        with zipfile.ZipFile(import_path, mode="r") as zf:
            with zf.open("hcV3-stories.csv", mode="r") as f:
                corpus = pd.read_csv(f)
    else:
        raise ValueError(f"Unprepared for {corpus_id} corpus.")
    return corpus, metadata


def load_matplotlib_settings():
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["interactive"] = True
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.cal"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["axes.linewidth"] = 0.8 # edge line width
    plt.rcParams["axes.axisbelow"] = True
    # plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["axes.grid.which"] = "major"
    plt.rcParams["axes.labelpad"] = 4
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["grid.alpha"] = 1
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.title_fontsize"] = 8
    plt.rcParams["legend.borderpad"] = .4
    plt.rcParams["legend.labelspacing"] = .2 # the vertical space between the legend entries
    plt.rcParams["legend.handlelength"] = 2 # the length of the legend lines
    plt.rcParams["legend.handleheight"] = .7 # the height of the legend handle
    plt.rcParams["legend.handletextpad"] = .2 # the space between the legend line and legend text
    plt.rcParams["legend.borderaxespad"] = .5 # the border between the axes and legend edge
    plt.rcParams["legend.columnspacing"] = 1 # the space between the legend line and legend text
