"""Convert a raw dream corpus to a standardized format.

Each corpus is handled differently, with most info
coming from the config.json configuration file.

Saves a single zipped folder full of txt files.
Each dream report is a single txt file with a name
following a BIDS-ish format of:
    corp-dreamviews_auth-1_text-2.txt
    (First author of the dreamviews corpus, second dream from that author)
"""

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import utils


def remove_nas_and_empties(df, text_column):
    df = df.copy().dropna(axis="columns", how="all")
    df = df.dropna(subset=text_column)
    df = df[df[text_column].str.len().gt(0)]
    return df


def process_and_split(df, corpus_id, author_column, text_column, time_column=None):
    # First sort so text numbers are more appropriate.
    sort_by = [author_column]
    if time_column is not None:
        sort_by.append(time_column)
    df = df.copy().sort_values(by=sort_by)
    # Add columns with critical identifiers.
    unique_authors = df[author_column].unique().tolist()
    df.insert(
        0,
        "entry_id",
        df.groupby(author_column)[author_column]
        .transform(lambda ser: range(ser.size))
        .add(1)
        .map(lambda x: f"entr-{x:d}"),
    )
    df.insert(
        0,
        "author_id",
        df[author_column]
        .map(lambda x: unique_authors.index(x))
        .add(1)
        .map(lambda x: f"auth-{x:d}"),
    )
    df.insert(0, "corpus_id", f"corp-{corpus_id:s}")
    ### THIS APPLY LINE TAKES A LONG TIME AND LOTS OF MEMORY. DUMB.
    df["text_id"] = df.apply(
        lambda row: "_".join(row[["corpus_id", "author_id", "entry_id"]]), axis=1
    )
    df["corpus_id"] = df["corpus_id"].str.split("-").str[1]
    df["author_id"] = df["author_id"].str.split("-").str[1].astype(int)
    df["entry_id"] = df["entry_id"].str.split("-").str[1].astype(int)
    attributes_df = df.set_index("text_id")
    text_ser = attributes_df.pop(text_column)
    return text_ser, attributes_df


def export_zip(
    text: pd.Series,
    attr: pd.DataFrame,
    metadata: dict,
    filepath: str,
    mode="x",
):
    """Export a pandas Series of text objects as individual txt files."""
    assert text.notna().all()
    assert text.str.len().gt(0).all()
    assert attr["corpus_id"].nunique() == 1
    corpus_id = attr["corpus_id"].unique()[0]
    if not isinstance(filepath, Path):
        assert isinstance(filepath, str)
        filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filepath, mode=mode) as zf:
        zf.writestr(f"{corpus_id}.json", json.dumps(metadata, indent=4))
        zf.writestr(f"{corpus_id}.tsv", attr.to_csv(sep="\t", index=True, na_rep="n/a"))
        for text_id, text_str in tqdm(text.items(), total=text.size, desc=corpus_id):
            # text_bytes = text_str.encode("utf-8")
            zf.writestr(f"{text_id}.txt", text_str)


# def get_column_name(category, column_info):
#     legend = {
#         "author": "Unique author identifier",
#         "text": "Entry text",
#         "timestamp": "Datetime of entry",
#         "corpus": "Unique corpus identifier",
#     }
#     target_str = legend[category]
#     opts = [ k for k, v in column_info.items() if v.startswith(target_str) ]
#     if opts:
#         assert len(opts) == 1
#         return opts[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", type=str, required=True, choices=utils.config["corpora"])
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite standard file output if it already exists.",
    )
    args = parser.parse_args()

    corpus_id = args.corpus
    write_mode = "w" if args.overwrite else "x"

    deriv_dir = Path(utils.config["derivatives_directory"])
    export_dir = deriv_dir / "zips"
    export_name = f"{corpus_id}.zip"
    export_path = export_dir / export_name

    # Load corpus as a pandas dataframe.
    corpus, metadata = utils.load_sourcedata(corpus_id)

    author_column = metadata["AuthorColumn"]
    text_column = metadata["TextColumn"]
    time_column = metadata["TimeColumn"] if "TimeColumn" in metadata else None

    # Clean up idiosyncratic errors.
    if corpus_id == "thoughtpings":
        corpus["valence"] = corpus["valence"].replace(
            {
                "A 9 very positive ": 9,
                "9 Very positive ": 9,
                "98": 8,
                "I am feeling valence at an 8 right now": 8,
                "I was thinking about the exam I was taking for my research methods in psychology class as I just finished. ": pd.NA,
                "&": pd.NA,
                "Positive ": pd.NA,
            }
        )
        corpus["valence"] = pd.to_numeric(corpus["valence"], errors="raise", downcast="integer")
        corpus["arousal"] = corpus["arousal"].replace(
            {
                "3 calm ": 3,
                "Around 7": 7,
                "9 very active ": 9,
                "^": pd.NA,
            }
        )
        corpus["arousal"] = pd.to_numeric(corpus["arousal"], errors="raise", downcast="integer")
        corpus["off_task"] = corpus["off_task"].replace(
            {
                "y": "Y",
                "n": "N",
                "Y ": "Y",
                "N ": "N",
                "  N": "N",
                "B": pd.NA,
                "M": pd.NA,
                "T": pd.NA,
                "5": pd.NA,
                "6": pd.NA,
                "Arousal:6\nThinking: Y": pd.NA,
                "I was thinking about working out after I get out of the class I'm currently in ": pd.NA,
            }
        )
        ## !! note could just coerce the irregularities into NA but i like being explicit
        corpus["wandering"] = corpus["wandering"].replace(
            {
                "Y": pd.NA,
                "N": pd.NA,
                "I'm thinking about whether or not I should go y": pd.NA,
                "I was thinking about how I should stay awake and pay attention to class but also how sleepy I am and how nice and comfortable it would be to close my eyes": pd.NA,
                "I'm thinking about the show I'm watching ": pd.NA,
                "I'm in class trying to think of what song by Clairo is stuck in my head right now so I was trying to go through the entire song until I remembered the name ": pd.NA,
                "The exam about tomorrow ": pd.NA,
                "I was thinking about how bad my acne is and how it makes me want to die and cut my skin off": pd.NA,
                "Why did joe just get arrested? The police are huge hardos I hate them ": pd.NA,
                "I'm taking a picture of my shoes to send to my parents, but I can't get the lighting right, so I open the window and when I look back at my phone, a friend has messages me about some plans tonight ": pd.NA,
            }
        )
        corpus["wandering"] = pd.to_numeric(corpus["wandering"], errors="raise", downcast="integer")

    # Export as zipfile of txts.
    corpus = remove_nas_and_empties(corpus, text_column)
    # if corpus_id == "thinkaloud":
    #     # String individual thought segments into one "thought" per entry.
    #     # (losing some other potentially interesting attributes in the process)
    #     corpus = corpus.groupby([author_column, "TrialNumber"]
    #         )[text_column].agg(lambda s: " ".join(s)
    #         ).reset_index()
    text_ser, attr_df = process_and_split(
        corpus, corpus_id, author_column, text_column, time_column
    )
    export_zip(text_ser, attr_df, metadata, export_path, mode=write_mode)
