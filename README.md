# Brain Language Model

A research pipeline for learning discrete representations of brain activity and training an autoregressive Transformer over the resulting brain-token sequences.

## Pipeline

```text
DS003844 BrainVision recordings
        |
        v
Preprocessing + 5-second windows (50% overlap)
        |
        v
Adaptive brain graphs (pairwise Mutual Information)
        |
        v
GAT graph autoencoder
        |
        v
128-dimensional spatial graph embeddings
        |
        v
Spatio-temporal Transformer embeddings
        |
        v
Residual Vector Quantization (RVQ) tokenizer sweep
        |
        v
Discrete brain-token sequences
        |
        v
Autoregressive Brain Language Model
        |
        v
Held-out next-token loss / perplexity / top-k diagnostics
```

## Repository structure

```text
brain-language-model/
├── notebooks/
│   ├── 01_preprocessing_adaptive_brain_graphs.ipynb
│   ├── 02_graph_autoencoder_gat.ipynb
│   ├── 03_spatiotemporal_embeddings_gat.ipynb
│   ├── 04_rvq_tokenizer_sweep.ipynb
│   └── 05_brain_language_model_training.ipynb
├── data/
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The notebooks were developed around the DS003844 dataset, using BrainVision `.vhdr` recordings and corresponding `_channels.tsv` channel metadata.

The preprocessing notebook:
- discovers `.vhdr` and `_channels.tsv` files recursively;
- retains recordings whose sampling frequency is at least 2048 Hz;
- removes channels marked as `bad` in the channel metadata;
- applies a 50 Hz notch filter and resamples to 1024 Hz;
- creates 5-second windows with 50% overlap;
- assigns `PRE`, `INTER`, `POST`, or `POST+` labels from the recording filename;
- constructs recording metadata and adaptive graphs using pairwise Mutual Information.

The project notes associated with this work report 33 retained recordings, 6 subjects, 18 sessions, and 3,684 windows. Re-run preprocessing against your local copy of the dataset to verify these counts.

**The dataset and generated intermediate artifacts are not included in this repository.** Obtain the dataset from its official source and follow its access conditions and data-use terms. Do not commit restricted data, raw recordings, or large model checkpoints.

## Notebook order and generated artifacts

Run the notebooks in order. They currently contain Google Colab / Google Drive paths under `/content/drive/MyDrive/ds003844_processed`; update those paths to your own storage location before running outside the original Colab setup.

| Order | Notebook | Main outputs expected by the next stage |
|---|---|---|
| 1 | `01_preprocessing_adaptive_brain_graphs.ipynb` | Processed `.fif` recordings, `recording_metadata.pkl`, `adaptive_graph_batches/*.pkl` |
| 2 | `02_graph_autoencoder_gat.ipynb` | `graph_encoder.pt`, `spatial_graph_embeddings.pt` |
| 3 | `03_spatiotemporal_embeddings_gat.ipynb` | `spatio_temporal_embeddings.pt`, `spatio_temporal_metadata.pkl` |
| 4 | `04_rvq_tokenizer_sweep.ipynb` | RVQ checkpoints and, if configurations pass the funnel, `rvq_sweep/shortlist_for_blm.pkl` |
| 5 | `05_brain_language_model_training.ipynb` | BLM checkpoints, plots, summary, and token sequence CSV |

The RVQ notebook is designed to screen tokenizer configurations before spending compute on BLM training. The BLM notebook expects `rvq_sweep/shortlist_for_blm.pkl` to exist.

### Important reproducibility note

The saved outputs embedded in the supplied RVQ sweep notebook report **0 of 10 configurations passing its funnel**, and therefore no shortlist being saved. The supplied BLM training notebook's saved outputs, however, report a completed run using a shortlist and identify `cb16_L1` as its best configuration. These outputs may come from different runs or different versions of intermediate files. They should not be presented as one reproducible end-to-end experiment without rerunning the pipeline and preserving the exact inputs, configuration, and outputs for each run.

The supplied notebooks have had their cell outputs cleared in this repository copy to keep the notebooks lightweight and avoid publishing stale execution results. The source code and markdown cells are retained.

## Environment

The notebooks use:
- Python
- NumPy, pandas, Matplotlib
- MNE-Python
- scikit-learn and joblib
- PyTorch
- PyTorch Geometric
- tqdm

Install the general dependencies:

```bash
pip install -r requirements.txt
```

Install **PyTorch and PyTorch Geometric** using the official installation selector for your operating system, Python version, and CUDA version. The correct PyTorch Geometric wheels depend on the installed PyTorch build. See:
- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

The notebooks were run in Google Colab with GPU acceleration for the deep-learning stages. CPU execution may be considerably slower.

## Data and artifact layout

The original notebooks expect a working directory similar to:

```text
ds003844_processed/
├── recording_metadata.pkl
├── adaptive_graph_batches/
├── graph_autoencoder/
├── graph_encoder.pt
├── spatial_graph_embeddings.pt
├── spatio_temporal_embeddings.pt
├── spatio_temporal_metadata.pkl
├── rvq_sweep/
└── blm/
```

Keep generated artifacts outside Git unless they are small, intentionally versioned research outputs. For large files, use an appropriate artifact store or Git LFS after checking dataset and model licensing.

## Research caveats

- The number of windows is not the number of independent recordings; windows overlap by 50%.
- The RVQ and BLM notebooks use recording-level train/validation splits to reduce leakage from adjacent overlapping windows.
- A low held-out perplexity alone does not establish that the model has learned clinically meaningful or subject-generalizable brain representations. Compare against simple baselines and report subject/session-level evaluation.
- Record the exact dataset version, split indices, random seeds, package versions, model checkpoints, and tokenizer configuration for every reported experiment.

## Citation and license

Add the dataset's required citation and usage terms before publishing. No software license is included yet; choose one only after deciding how you want others to reuse the code and confirming that the dataset and any included assets permit it.
