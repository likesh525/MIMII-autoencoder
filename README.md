# Pump-Net: Industrial Pump Anomaly Detection (Autoencoder)

This repository explores **unsupervised anomaly detection** on industrial pump sounds (Normal vs Abnormal) using a **dense autoencoder** trained on normal samples and evaluated using **reconstruction error (MSE)**.

It also explores the **possibility of using quantized models** (e.g., TFLite) to achieve similar anomaly-detection performance with lower model size and faster inference.

## Where to look first


- **working/consolidated notebook:** [autoencoder_draft.ipynb](autoencoder_draft.ipynb)  [Refer this]
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

Top‑N selection rule: After training an 18‑D autoencoder, we compute per‑feature reconstruction MSE on an evaluation split `(X_combined_test = [normal val] + [abnormal test]),` rank features by this per‑feature MSE (descending), and keep the top 5 indices. We then slice the original feature matrix to those 5 columns and train/evaluate separate 5‑D models

## Quantization
**TFLite quantization (what kind?):** 

The notebook uses post‑training full‑integer INT8 quantization via `tf.lite.TFLiteConverter with optimizations = [tf.lite.Optimize.DEFAULT], supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8], and inference_input_type = inference_output_type = tf.int8.`

 A representative (calibration) dataset is provided from a small slice of the scaled normal training data (first ~100–200 samples) using a generator (`representative_data_gen*`). Features are scaled with `MinMaxScaler(feature_range=(-1, 1))` for quantization experiments.

## Saved artifacts

Trained models are not included in the repo:

- Keras autoencoders: `large_ae_*.keras`, `small_ae_*.keras`, `shallow_ae_*.keras`
- TFLite exports: `small_model_*.tflite`, `shallow_model_*.tflite`

## Key results (from the benchmark tables in [autoencoder_draft.ipynb](autoencoder_draft.ipynb))

### Best overall accuracy (Keras)

| Model | Input dims | AUROC | AUPRC |
|---|---:|---:|---:|
| `large_ae_full_features` | 18 | 0.99172 | 0.99529 |
| `shallow_ae_full_features` | 18 | 0.98673 | 0.99265 |
| `small_ae_full_features` | 18 | 0.98014 | 0.98991 |

**Finding:** Using the **full 18-feature** input consistently outperformed using only the 5-feature subset.

### Top-N feature subset (Keras)

| Model | Input dims | AUROC | AUPRC |
|---|---:|---:|---:|
| `large_ae_topN_features` | 5 | 0.97553 | 0.98935 |
| `shallow_ae_topN_features` | 5 | 0.97205 | 0.98796 |
| `small_ae_topN_features` | 5 | 0.97271 | 0.98823 |

### Edge-style deployment baseline (TFLite)

| Model | Input dims | AUROC | AUPRC | Mean latency (ms/batch) |
|---|---:|---:|---:|---:|
| `small_model(tflite)_full_features` | 18 | 0.38274 | 0.70511 | 1.57232 |
| `small_model(tflite)_topN_features` | 5 | 0.08771 | 0.45763 | 1.449385 |
| `shallow_model(tflite)_full_features` | 18 | 0.92509 | 0.996783 | 1.546835 |
| `shallow_model(tflite)_topN_features` | 5 | 0.324299 | 0.456364 | 1.550580 |

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
- TopN feature selection is problematic, data leakage needs to be corrected