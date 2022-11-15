"""
Write to a tsv on-the-fly to avoid memory issues.
"""
import argparse
import csv
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm import tqdm

import utils



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--corpus", type=str, required=True, choices=utils.config["corpora"])
parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite standard file output if it already exists.")
args = parser.parse_args()

corpus_id = args.corpus
write_mode = "wt" if args.overwrite else "x"

source_dir = Path(utils.config["source_directory"])
deriv_dir = Path(utils.config["derivatives_directory"])
import_path = deriv_dir / "zips" / f"{corpus_id}.zip"
export_path = deriv_dir / f"corp-{corpus_id}_scores.tsv"

id_headers = ["corpus_id", "author_id", "entry_id"]
length_metrics = [
    "Ncharacters",
    "Ntokens",
    "Nwords",
    "Nnounchunks",
    "Nsentences",
]
coherence_metrics = [
    "CoherenceN",
    "CoherenceMean",
    "CoherenceVar",
    "CoherenceMin",
    "CoherenceMax",
]

column_names = id_headers + length_metrics + coherence_metrics

batch_size = 500
pbar_kwargs = dict(desc=f"Coherence {corpus_id}")
pipe_kwargs = dict(batch_size=batch_size, as_tuples=True, n_process=1)

lang_model = "en_core_web_lg"

# token filter for coherence analysis
def token_filter(token):
    return (token.has_vector and token.is_alpha and token.pos_ in ["NOUN", "ADJ", "VERB"]
        ) and not (token.is_stop or len(token) <= 3 or token.like_num)


# Open a file to write results to line-by-line.
with open(export_path, mode=write_mode, encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile, delimiter="\t", quoting=csv.QUOTE_NONE)
    
    # Write column header.
    writer.writerow(column_names)

    # Load spacy model.
    nlp = spacy.load(lang_model)
    pipe_kwargs["disable"] = ["lemmatizer", "ner"]

    with zipfile.ZipFile(import_path, mode="r") as zf:
        txt_files = [ n for n in zf.namelist() if n.endswith(".txt") ]
        file_gen = ( (zf.read(fn).decode("utf-8"), utils.parse_text_id(fn)) for fn in txt_files )
        pbar_kwargs.update(dict(total=len(txt_files)))


        # #### hack to save out full vector and text chunks
        # #### for visual inspection of text and ability to look at semantic timecourse
        # distance_vects = {}
        # chunks_text = {}


        # Loop over each document.
        for doc, context in tqdm(nlp.pipe(file_gen, **pipe_kwargs), **pbar_kwargs):

            length_scores = {
                "Ncharacters": len(doc.text),
                "Ntokens": len(doc),
                "Nwords": len([ t for t in doc if t.is_alpha ]),
                "Nnounchunks": len(list(doc.noun_chunks)),
                "Nsentences": len(list(doc.sents)),
            }

            # if length_scores["Nwords"] < 10:
            #     continue

            length_scores_list = [ length_scores[m] for m in length_metrics ]

            ## Coherence section

            vectors = [] # to save summary vectors of each noun chunk
            # chunk_strings = [] # to save each noun chunk

            # Loop over each noun chunk of the current document.
            # for n in doc:
            if corpus_id == "thoughtpings":
                phrases = doc.noun_chunks
            else:
                phrases = doc.sents
            for n in phrases:
                # Check if more than one noun chunk passes token filtering.
                if (subgroup_vectors := [ t.vector for t in n if token_filter(t) ]):
                # if (subgroup_vectors := [ t.vector for t in n if t.has_vector ]):
                    # If so, get an average vector for this noun chunk and save it.
                    chunk_vect = np.row_stack(subgroup_vectors).mean(axis=0)
                    vectors.append(chunk_vect)
                # if token_filter(n):
                #     vectors.append(n.vector)
                # if (subgroup := [ t for t in n if token_filter(t) ]):
                #     subgroup_vectors = [ t.vector for t in subgroup ]
                #     subgroup_text = [ t.text for t in subgroup ]
                #     vectors.append(np.row_stack(subgroup_vectors).mean(axis=0))
                #     chunk_strings.append(subgroup_text)
            
            # Get coherence across the chunks, if long enough.
            if len(vectors) > 1:

                # Stack vectors and get similarities
                arr = np.row_stack(vectors)
                w2w = cosine_similarity(arr).diagonal(offset=1)
                # w2w = np.array([ distance.cosine(x,y) for x, y in zip(arr[1:], arr[:-1]) ])

                coh_scores = {}
                coh_scores["CoherenceN"] = w2w.size # number of coherence values (or 1 less than number of chunks)
                coh_scores["CoherenceMean"] = w2w.mean()
                coh_scores["CoherenceVar"] = w2w.var()
                coh_scores["CoherenceMin"] = w2w.min()
                coh_scores["CoherenceMax"] = w2w.max()
                coh_scores_list = [ coh_scores[m] for m in coherence_metrics ]

                datarow = context + length_scores_list + coh_scores_list
                assert len(datarow) == len(column_names)
                writer.writerow(datarow)

                # tc_key = "-".join(context)
                # distance_vects[tc_key] = w2w
                # chunks_text[tc_key] = chunk_strings

        # # Save distance timecourses as numpy.
        # export_path2 = deriv_dir / f"corp-{corpus_id}_scor-{scores_id}.npz"
        # np.savez_compressed(export_path2, **distance_vects)
        # # Save texts as json file
        # export_path3 = deriv_dir / f"corp-{corpus_id}_scor-{scores_id}.json"
        # dmlab.io.export_json(chunks_text, export_path3, mode=write_mode)
