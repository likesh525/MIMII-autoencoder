# Pump-Net: Industrial Pump Anomaly Detection (Autoencoder)

This repository explores **unsupervised anomaly detection** on industrial pump sounds (Normal vs Abnormal) using a **dense autoencoder** trained on normal samples and evaluated using **reconstruction error (MSE)**.

It also explores the **possibility of using quantized models** (e.g., TFLite) to achieve similar anomaly-detection performance with lower model size and faster inference.

## Where to look first

- **Working / exploratory notebook:** [autoencoder.ipynb](autoencoder.ipynb)
- **More consolidated notebook:** [autoencoder_draft.ipynb](autoencoder_draft.ipynb)  [Refer this]
- **Dependencies:** [requirements.txt](requirements.txt)

## Data layout

The notebooks expect the dataset under:

- `data/normal/`
- `data/abnormal/`

## Approach (high level)

1. Extract audio features.
2. Train an autoencoder on **normal** data only.
3. Score test samples using reconstruction **MSE**.
4. Evaluate with ranking metrics (**AUROC**, **AUPRC**) and benchmark inference latency.

## Feature sets tested

Two input configurations are compared:

- **Full features (18 dims)**
- **Top-N features (5 dims)** (feature selection done in the notebook)

## Saved artifacts

Trained models are not included in the repo:

- Keras autoencoders: `large_ae_*.keras`, `small_ae_*.keras`, `shallow_ae_*.keras`
- TFLite exports: `small_model_*.tflite`, `shallow_model_*.tflite`

## Key results (from the benchmark tables in [autoencoder_draft.ipynb](autoencoder_draft.ipynb))

### Best overall accuracy (Keras)

| Model | Input dims | AUROC | AUPRC |
|---|---:|---:|---:|
| `large_ae_full_features` | 18 | 0.986637 | 0.992548 |
| `shallow_ae_full_features` | 18 | 0.974873 | 0.986584 |
| `small_ae_full_features` | 18 | 0.972991 | 0.986404 |

**Finding:** Using the **full 18-feature** input consistently outperformed using only the 5-feature subset.

### Top-N feature subset (Keras)

| Model | Input dims | AUROC | AUPRC |
|---|---:|---:|---:|
| `large_ae_topN_features` | 5 | 0.947770 | 0.973597 |
| `shallow_ae_topN_features` | 5 | 0.942500 | 0.971342 |
| `small_ae_topN_features` | 5 | 0.941464 | 0.971155 |

### Edge-style deployment baseline (TFLite)

| Model | Input dims | AUROC | AUPRC | Mean latency (ms/batch) |
|---|---:|---:|---:|---:|
| `small_model(tflite)_full_features` | 18 | 0.453887 | 0.661609 | 1.528710 |
| `small_model(tflite)_topN_features` | 5 | 0.316582 | 0.665958 | 1.479690 |
| `shallow_model(tflite)_full_features` | 18 | 0.860437 | 0.941244 | 1.676835 |
| `shallow_model(tflite)_topN_features` | 5 | 0.324299 | 0.676364 | 1.550580 |

**Finding:** TFLite inference is ~1.5–1.7 ms per batch in this desktop benchmark. Accuracy depends heavily on the architecture and feature set: the **shallow full-features TFLite** (`shallow_model(tflite)_full_features`) model retained strong AUROC, while the **small TFLite** variants performed poorly.

## Reproducing

1. Create/activate a Python 3.10 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the working notebook:
   - [autoencoder_draft.ipynb](autoencoder_draft.ipynb)

## Notes

- Benchmarks were recorded on a **desktop runtime** (not on physical MCU hardware).
- Some model naming in tables may include minor typos (e.g., a trailing `d` in one TFLite model label inside the notebook output). The corresponding `.tflite` filenames in the repo are the authoritative artifacts.
