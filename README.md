## KeyClass Reproduction Project
### Jonathan Luo (jluo28@illinois.edu)

Original Paper: [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088)

## Environment Setup

Setup the environment with the following steps: 

``` bash
$ conda create -n keyclass python=3.8
$ conda activate keyclass
$ conda install -c pytorch pytorch=1.10.0 cudatoolkit=11.3
$ conda install -c conda-forge snorkel=0.9.8
$ conda install -c huggingface tokenizers=0.10.3
$ conda install -c huggingface transformers=4.11.3
$ conda install -c conda-forge sentence-transformers=2.2.1
$ conda install -c labscript-suite windows-curses
$ conda install pandas
$ conda install jupyter notebook
```

To run the experiment, you can either go through the cells in `final.ipynb`, or just run `python run_experiment.py`. 

## Data setup

1. Create a folder `data` in the root directory of the project
2. Download the DIAGNOSES_ICD and NOTEEVENTS tables from the [MIMIC III dataset](https://physionet.org/content/mimiciii/1.4/) and place them in the newly created `data` folder.
3. Download the "Version 32 Full and Abbreviated Code Titles  – Effective October 1, 2014 (ZIP)" zip from [cms.gov's official list of ICD9 codes](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles)
4. Place the `CMS32_DESC_LONG_DX` document from the zip into the `data` folder.


## Relevant Files/Directories

- `final.ipynb`: iPython notebook with step-by-step outputs for the final multi-label experiment on the MIMIC III data.
- `run_experiment.py`: Essentially the same as `final.ipynb`, but in Python script format
- `imdb_test.ipynb`: iPython notebook with step-by-step outputs for the experiment on the baseline imdb dataset.
- `mine_descriptions`: Separate code for mining class descriptions for the ICD9 codes, although this is also included in `final.ipynb`.
- `KeyClass-single-label/`: Single label implementation of KeyClass. Essentially the same as the code given by the authors.
- `KeyClass/`: Multi label implementation of KeyClass.

## Outputs

n-gram embeddings will be saved in `data/mimic/`. The label model will be saved in `data/results/`.

## Results Overview

| Method                 | Recall               | Precision            | F1 Score             |
|------------------------|----------------------|----------------------|----------------------|
| Weakly Supervised FasTag | 0.734 ± 0.00138      | 0.436 ± 0.00144      | 0.525 ± 0.00133      |
| Author's KeyClass        | 0.896 ± 0.0009       | 0.507 ± 0.0016       | 0.6252 ± 0.0014      |
| My KeyClass              | 0.442 ± 0.0488       | 0.475 ± 0.0016       | 0.457 ± 0.0278       |
