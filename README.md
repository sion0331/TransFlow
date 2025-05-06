# DeepTransLOB

This project implements and extends the **TransLOB** architecture for high-frequency limit order book (LOB) prediction, with support for both the **FI-2010** dataset and high-frequency **crypto** data (e.g., BTC/USD). It includes:

- Causal convolutions to preserve time-ordering
- Inception-style modules (optional)
- Transformer-based temporal modeling
- Normalization and preprocessing pipelines
- Model evaluation and visualization

## Project Structure
.
├── README.md
├── Untitled.ipynb
├── data
│   ├── FI-2010
│   └── crypto
├── main_results.ipynb
├── models
│   ├── DeepTransLOB
│   ├── __pycache__
│   ├── deepLOB
│   ├── deep_lob.py
│   ├── deep_trans_lob.py
│   ├── transLOB
│   └── trans_lob.py
├── outputs
│   ├── crypto
│   └── fi2010
├── requirements.txt
├── train_crypto.ipynb
├── train_fi2010.ipynb
└── utils
    ├── __pycache__
    ├── crypto_loader.py
    ├── fi2010_loader.py
    ├── plots.py
    ├── preprocessing.py
    └── training.py

14 directories, 14 files
