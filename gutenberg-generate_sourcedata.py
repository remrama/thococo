"""
Generate gutenberg source data from Standardized Project Gutenberg corpus,
which is HUGE and we only want like 5-10 books.

Also we need to shuffle sentences around a few times.
"""
from copy import copy
from pathlib import Path
import random

from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm

import utils


random.seed(9)

gutenberg_dir = Path("../../../packages/gutenberg")
source_dir = Path(utils.config["source_directory"])
import_path = gutenberg_dir / "metadata" / "metadata.csv"
export_path = source_dir / "gutenberg.tsv"

book_ids = [
    "PG11", # Alice in Wonderland
    "PG10", # King James Bible
    "PG2701", # Moby Dick
    "PG26", # Paradise Lost
    "PG1342", # Pride and Prejudice
    "PG84", # Frankenstein
    "PG345", # Dracula
    "PG43", # The Strange Case of Dr. Jekyll and Mr. Hyde
    "PG64317", # The Great Gatsby
    "PG219", # Heart of Darkness
    "PG205", # Walden, and On The Duty Of Civil Disobedience
    "PG16", # Peter Pan
    "PG3296", # The Confessions of St. Augustine
    "PG36", # The War of the Worlds
    "PG1228", # On the Origin of Species
    "PG2680", # Meditations
    "PG5230", # The Invisible Man: A Grotesque Romance
    "PG10615", # An Essay Concerning Humane Understanding
    "PG932", # The Fall of the House of Usher
    "PG45315", # The Marriage of Heaven and Hell
    "PG164", # Twenty Thousand Leagues under the Sea
    "PG10148", # The Merry Adventures of Robin Hood
]

df = pd.read_csv(import_path)

df = df.query(f"id.isin({book_ids})")

def get_book_text(bookid):
    try:
        text_path = gutenberg_dir / "data" / "text" / f"{bookid}_text.txt"
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return pd.NA

df["rawtext"] = df["id"].map(get_book_text)


df = df.dropna(subset="rawtext")

df["nchars"] = df["rawtext"].str.len()

df = df.query("nchars.between(100000, 1000000)")

df = df[["id", "title", "author", "rawtext"]]


def shuffle_sentences(txt, shuffle_rate):
    sentences = sent_tokenize(txt)
    n_total_sentences = len(sentences)
    n_shuffled_sentences = int(n_total_sentences*shuffle_rate)
    shuffle_indices = random.sample(range(n_total_sentences), k=n_shuffled_sentences)
    shuffled_sentences = [ sentences[i] for i in shuffle_indices ]
    random.shuffle(shuffled_sentences)
    for index, random_sentence in zip(shuffle_indices, shuffled_sentences):
        sentences[index] = random_sentence
    return " ".join(sentences)

shuffle_rates = [0, .1, .2, .3, .5, .7, 1]
for r in tqdm(shuffle_rates, desc="Shuffle rates"):
    df[f"shuf-{r}"] = df["rawtext"].apply(shuffle_sentences, shuffle_rate=r)


shuffled_columns = [ c for c in df if c.startswith("shuf") ]

df_long = df.melt(value_vars=shuffled_columns, id_vars=["id", "author", "title"],
    var_name="shuffle_rate", value_name="text")

df_long["shuffle_rate"] = df_long["shuffle_rate"].str.split("-").str[1].astype(float)

df_long = df_long.sort_values(["author", "shuffle_rate"])

df_long.to_csv(export_path, index=False, sep="\t")