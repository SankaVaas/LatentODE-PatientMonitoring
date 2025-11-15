# Neural ODE for ICU Patient Monitoring

## Project Overview
Predicting adverse events (mortality, sepsis) in ICU patients using Neural Ordinary Differential Equations (Neural ODEs) for continuous-time modeling of patient vitals.

## Key Features
- Handles irregular time series sampling
- Continuous-time patient state modeling
- Uncertainty quantification
- Multi-variate vital signs integration

## Dataset
Using MIMIC-III Demo dataset (publicly available)
- 100 ICU patients
- Time-series vitals: Heart Rate, Blood Pressure, SpO2, Temperature, etc.

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── data/
│   ├── raw/          # Raw MIMIC-III data
│   └── processed/    # Preprocessed data
├── src/              # Source code
├── models/           # Saved models
├── results/          # Outputs and plots
└── notebooks/        # Jupyter notebooks
```

## Steps
1. [DONE] Setup and data preparation
2. Data preprocessing and feature engineering
3. Neural ODE model implementation
4. Training pipeline
5. Evaluation and visualization
