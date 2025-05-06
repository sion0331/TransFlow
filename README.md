# DeepTransLOB: Enhancing Transformer-Based Models for Limit Order Book Prediction
This repository contains code to replicate and extend the DeepLOB and TransLOB architectures for limit order book (LOB) data. We experiment on both the FI-2010 benchmark dataset and high-frequency crypto (BTC/USD) data.

1. Install Dependencies
- pip install -r requirements.txt

2. Prepare Datasets
- Subsets of the original datasets (with _MINI suffix) are already included in this repo for quick testing
- FI-2010: Download the full dataset and place it under ./data/FI-2010/
    - https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Crypto (BTC/USD): Download and place BTC_1sec.csv under ./data/crypto/
    - https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data

3. Train Models
- FI-2010: train_fi2010.ipynb
- Crypto LOB: train_crypto.ipynb
- Model weights will be saved as .pth files under ./outputs
- Training history and validation metrics will be saved as .pkl files under ./outputs

4. View Results
- main_results.ipynb


## Project Structure
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── main_results.ipynb         # Plotting and model comparison
├── train_fi2010.ipynb         # FI-2010 training pipeline
├── train_crypto.ipynb         # Crypto training pipeline
├── data/
│   ├── FI-2010/               # FI-2010 dataset (organized by normalization)
│   └── crypto/                # Raw crypto CSV data
├── models/
│   ├── deep_lob.py            # DeepLOB implementation (unchanged from original paper)
│   ├── trans_lob.py           # TransLOB implementation (unchanged from original paper)
│   └── deep_trans_lob.py      # New extended model with 2D Convolutions + Transformer
├── outputs/
│   ├── fi2010/                # Saved FI-2010 model weights and training history
│   └── crypto/                # Saved Crypto model weights and training history
└── utils/
    ├── crypto_loader.py       # Crypto dataset loader
    ├── fi2010_loader.py       # FI-2010 dataset loader
    ├── preprocessing.py       # Feature engineering, labeling, normalization
    ├── training.py            # Train/validate functions for models
    └── plots.py               # Label distributions and training curves
    
14 directories, 14 files
