# thococo

A research project testing how **tho**ught **co**ntents are **co**nstrained by cognitive control. The question is largely driven by the Dynamic Framework of Thought proposed in [Christoff et al., 2016, _Nat Rev Neurosci_](https://doi.org/10.1038/nrn.2016.113).


- `environment.yaml` can be used to construct the Python environment
- `config.json` has general parameter options that apply to multiple scripts
- `utils.py` has general functions that are useful to multiple scripts

```bash
# Download spaCy model.
python -m spacy download en_core_web_lg

# Generate data directory structure.
python setup_directories.py                     #=> derivatives/
                                                #=> derivatives/zips

# Convert raw data to standardized text files.
python source2txt.py --corpus thoughtpings      #=> zips/thoughtpings.zip
python source2txt.py --corpus hippocorpus       #=> zips/hippocorpus.zip
python source2txt.py --corpus dreamviews        #=> zips/dreamviews.zip

# Get semantic coherence scores.
python coherence.py --corpus thoughtpings       #=> corp-thoughtpings_scores.tsv
python coherence.py --corpus hippocorpus        #=> corp-hippocorpus_scores.tsv
python coherence.py --corpus dreamviews         #=> corp-dreamviews_scores.tsv

# Visualize sample sizes.
python agg-scores.py                            #=> agg-auth_scores.tsv & agg-corp_scores.tsv
python plot-samplesize.py                       #=> agg-corp_samplesize.png/pdf

# Run binned comparison within each dataset.
python twoway.py --corpus thoughtpings          #=> corp-thoughtpings_2way.tsv/png/pdf
python twoway.py --corpus hippocorpus           #=> corp-hippocorpus_2way.tsv/png/pdf
python twoway.py --corpus dreamviews            #=> corp-dreamviews_2way.tsv/png/pdf

# Correlate coherence with self-reported mind-wandering in the thoughtpings dataset.
python correlation.py                           #=> corp-thoughtpings_corr.tsv/png/pdf
```