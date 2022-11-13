# thococo

A research project testing how **tho**ught **co**ntents are **co**nstrained by cognitive control. The question is largely driven by the Dynamic Framework of Thought proposed in [Christoff et al., 2016, _Nat Rev Neurosci_](https://doi.org/10.1038/nrn.2016.113).


### General files

- `environment.yaml` can be used to construct the Python environment
- `config.json` has general parameter options that apply to multiple scripts
- `utils.py` has general functions that are useful to multiple scripts

```bash
# Download spaCy model.
python -m spacy download en_core_web_lg

# Generate data directory structure.
python setup_directories.py                     #=> derivatives/
                                                #=> derivatives/zips

python source2txt.py --corpus thinkaloud        #=> zips/thinkaloud.zip
python source2txt.py --corpus thoughtpings      #=> zips/thoughtpings.zip
python source2txt.py --corpus hippocorpus       #=> zips/hippocorpus.zip
python source2txt.py --corpus dreamviews        #=> zips/dreamviews.zip
```

```bash
python calc-scores.py -c thinkaloud             #=> corp-thinkaloud_scores.tsv
python calc-scores.py -c thoughtpings           #=> corp-thoughtpings_scores.tsv
python calc-scores.py -c hippocorpus            #=> corp-hippocorpus_scores.tsv
python calc-scores.py -c dreamviews             #=> corp-dreamviews_scores.tsv

python agg-scores.py                            #=> agg-auth_scores.tsv
                                                #=> agg-corp_scores.tsv
```

```bash
# plot sample size
python plot-samplesize.py                       #=> agg-corp_samplesize.png/pdf
```