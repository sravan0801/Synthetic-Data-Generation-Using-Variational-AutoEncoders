# Synthetic Data Generation Using Variational Autoencoders

Generating synthetic credit card transaction data with a Variational Autoencoder (VAE).

## Overview

This project trains a VAE on a real credit card transaction dataset and uses it to
generate synthetic transactions that follow the same statistical distribution as the
original data. Synthetic data of this kind is useful for fraud analytics work where
sharing or augmenting real transaction data is restricted by privacy concerns.

All code lives in a single notebook: [`SyntheticDataGenerationMain.ipynb`](SyntheticDataGenerationMain.ipynb).

## Approach

### 1. Data preprocessing

- Load `card_transaction.v1.csv` (the IBM TabFormer credit card transactions dataset,
  ~24M rows).
- Strip the `$` symbol from the `Amount` column and cast it to `float`.
- Fill missing values in `Errors?` with `"No Error"` and fill any remaining nulls with
  the column mode.
- Label-encode every categorical column with `LabelEncoder`.
- Scale all columns to `[0, 1]` with `MinMaxScaler`.

### 2. VAE architecture

| Component | Layers |
| --- | --- |
| Encoder | `Input(input_dim)` → `Dense(12, relu)` → `Dense(5, relu)` → `z_mean(5)`, `z_log_var(5)` |
| Sampling | reparameterization trick: `z = z_mean + exp(0.5 * z_log_var) * epsilon` |
| Decoder | `Input(5)` → `Dense(5, relu)` → `Dense(12, relu)` → `Dense(input_dim, sigmoid)` |

- Latent dimension: `5`
- Optimizer: `adam`
- Loss: `binary_crossentropy`

### 3. Training

- 80/20 train/test split (`random_state=42`).
- 50 epochs, batch size 32, with the test set used for validation.

### 4. Synthetic data generation

The trained encoder maps the real data to the latent space; latent samples are drawn
using `z_mean` and `z_log_var` plus Gaussian noise, then passed through the decoder and
inverse-transformed back to the original scale.

### 5. Evaluation

- **Visual:** overlaid histograms of real vs. synthetic values for each of the 15
  columns.
- **Quantitative:** mean squared error between the scaled real data and the scaled
  reconstruction. Reported MSE: **~0.0175**.

## Requirements

- Python 3
- `pandas`, `numpy`, `scikit-learn`, `tensorflow` / `keras`, `matplotlib`

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

## Usage

1. Download `card_transaction.v1.csv` and update the `pd.read_csv(...)` path in the
   first code cell (it currently points to a Google Drive path used on Colab).
2. Open the notebook and run all cells:

   ```bash
   jupyter notebook SyntheticDataGenerationMain.ipynb
   ```

## Contributors

- Vadapalli Sai Sravan (CS24MTECH02007)
- Supreet Shukla (CS24MTECH02004)
- Tarun Jangir (CS24MTECH02005)
- Taufique Ramzan Shaikh (CS24MTECH02006)
- Afzaal Ahmad (CS24MTECH02002)
