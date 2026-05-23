# Spatial Soil NPK Prediction Using Multispectral Data
### A Hybrid Geo-Spectral XGBoost Framework with Edge-Based Deployment Focus

> End-to-end ML pipeline for predicting soil Nitrogen, Phosphorus, and Potassium from Vis-NIR spectral data  
> **40,788 samples | 8,443 features | R²=0.950 (N) | Streamlit UI | Kriging Maps | GeoTIFF Export**  
> EC3201 — Department of ECE, IIIT Manipur (Sem 6)

---

## Overview

Traditional soil nutrient testing is slow, expensive, and spatially sparse. This project builds a machine learning framework that predicts soil NPK levels from multispectral Vis-NIR spectral data, eliminating the need for lab testing while generating continuous spatial nutrient maps.

The system processes raw spectral reflectance data through a structured pipeline — SNV normalization → Savitzky-Golay filtering → PLS dimensionality reduction → XGBoost prediction → Kriging spatial interpolation — and deploys a complete field-ready Streamlit interface supporting GeoTIFF and CSV inputs.

The architecture is intentionally designed to be **computationally lightweight for offline, edge-based deployment** in remote agricultural fields.

---

## Key Results

| Nutrient | Model | Metric | Score |
|---|---|---|---|
| Nitrogen (N) | XGBoost Regressor | R² | **0.950 ± 0.014** |
| Phosphorus (P) | XGBoost Classifier (4-class) | Accuracy | **43.9% ± 2.3%** |
| Potassium (K) | XGBoost Regressor | R² | **0.414 ± 0.083** |

- **Dataset:** LUCAS 2015 — 40,788 samples, 4,200 spectral bands (400–2500 nm)
- **Features engineered:** 8,443 (8,400 spectral + 43 chemistry/geographic)
- **Validation:** Group K-Fold (5 folds) with spatial separation — prevents data leakage
- **Training time:** ~11.4 minutes on standard laptop hardware

---

## Pipeline Architecture

```
Raw Multispectral Input (GeoTIFF / CSV)
        │
        ▼
Spectral Preprocessing
  ├── SNV Normalization      → removes multiplicative scatter
  ├── Savitzky-Golay (SG0)  → noise reduction
  └── 1st Derivative (SG1)  → enhances absorption peaks
        │
        ▼
Feature Engineering (8,443 total)
  ├── 8,400 Spectral (SG0 + SG1 stacked)
  ├── 13 Base Chemistry (pH, Clay, Silt, Sand, OC + interactions)
  ├── 2 K-Specific (Exchangeable Proxy, Weathering Index)
  └── 28 Geographic (One-hot country encoding)
        │
        ▼
PLS Dimensionality Reduction
  ├── N: 20 components
  ├── P: 25 components
  └── K: 25 components
        │
        ▼
XGBoost Models
  ├── N: Regressor (log1p transform)
  ├── P: Multi-class Classifier (4 fertility zones)
  └── K: Regressor (tuned, L1+L2 regularization)
        │
        ▼
Ordinary Kriging Interpolation → Continuous Spatial Maps
        │
        ▼
Streamlit UI → Interactive Maps + GeoTIFF 
```

---

## Model Details

### Nitrogen (N) — XGBoost Regressor
```
n_estimators: 320 | max_depth: 6 | learning_rate: 0.05
subsample: 0.8 | colsample_bytree: 0.7
```
Strong spectral correlation with Organic Carbon (OC) → highest R²

### Phosphorus (P) — XGBoost Classifier
```
n_estimators: 300 | max_depth: 10 | learning_rate: 0.1
objective: multi:softprob | num_class: 4
```
Converted to classification — P lacks direct spectral signature; discrete fertility zones more actionable for farmers

| Class | Range (mg/kg) |
|---|---|
| Low | 1.4 – 12.6 |
| Med-Low | 12.6 – 24.9 |
| Med-High | 24.9 – 44.4 |
| High | 44.4+ |

### Potassium (K) — XGBoost Regressor (Tuned)
```
n_estimators: 400 | max_depth: 8 | learning_rate: 0.03
subsample: 0.75 | reg_alpha: 0.2 | reg_lambda: 0.8
```
2 domain-specific engineered features:
- `K_Exchangeable_Proxy = Clay × pH × 0.1`
- `K_Weathering_Index = Sand / (Clay + 1)`

---

## Spatial Validation

Group K-Fold cross-validation with geographic separation ensures training and testing samples belong to **different EU regions** — preventing spatial data leakage and ensuring robust real-world generalization.

```
Fold 1: Train=32,564 | Test=8,224  → N R²=0.934 | P Acc=42.1% | K R²=0.402
Fold 2: Train=32,709 | Test=8,079  → N R²=0.948 | P Acc=42.9% | K R²=0.350
Fold 3: Train=32,641 | Test=8,147  → N R²=0.962 | P Acc=45.3% | K R²=0.508
Fold 4: Train=32,533 | Test=8,255  → N R²=0.966 | P Acc=42.2% | K R²=0.320
Fold 5: Train=32,564 | Test=8,224  → N R²=0.939 | P Acc=47.2% | K R²=0.488
─────────────────────────────────────────────────────────────────────────
Mean:                                  R²=0.950   Acc=43.9%   R²=0.414
```

---

## Results

| Plot | Description |
|---|---|
| ![N Regression](<img width="1920" height="1440" alt="Fig8_Regression_N" src="https://github.com/user-attachments/assets/442ba294-b36f-4004-a80b-1c63c0a7f82f" />
) | Nitrogen prediction vs actual |
| ![K Regression](<img width="1920" height="1440" alt="Fig9_Regression_K" src="https://github.com/user-attachments/assets/90812ab4-b688-4f28-8040-e1db6998813c" />
) | Potassium prediction vs actual |
| ![P Confusion](<img width="1920" height="1440" alt="Fig10_ConfusionMatrix_P" src="https://github.com/user-attachments/assets/c544da3e-0a99-407f-b81c-1774b524af76" />
) | Phosphorus 4-class confusion matrix |

---

## Streamlit UI Features

- **Input:** GeoTIFF (multispectral) or CSV with Lat/Lon columns
- **Live soil chemistry** fetched from SoilGrids API (pH, Clay, Silt, Sand, OC)
- **Geographic encoding** via OpenStreetMap reverse geocoding
- **Interactive orthomosaic** with NPK hover tooltips
- **Side-by-side NDVI map** (requires 4-band GeoTIFF with NIR)
- **Kriging-interpolated heatmaps** for N, P, K
- **RGB dominance map** (R=P, G=N, B=K)
- **Export:** GeoTIFF layers + CSV predictions

---

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Train Models
```bash
python src/train.py
# Output: p_fixed.pkl (production pipeline)
#         results/N_regression_plot.png
#         results/K_regression_plot.png
#         results/P_confusion_matrix.png
#         results/npk_fold_metrics.csv
```

### Run UI
```bash
streamlit run src/app.py
```
> **Note:** Download the trained model `p_fixed.pkl` from the link below and place it in the root directory before running the UI.

📦 **[Download Trained Model (p_fixed.pkl)](YOUR_DRIVE_LINK_HERE)**

### Data Setup
The LUCAS 2015 spectral dataset (~GB scale) is not included due to size.  
Download from: [European Commission JRC](https://esdac.jrc.ec.europa.eu/content/lucas-2015-topsoil-data)

Place spectral CSVs in:
```
LUCAS2015_spectra/LUCAS2015_Soil_Spectra_EU28/
```
The cleaned master dataset `lucas_npk_clean.csv` is included in `data/`.

---

## Repository Structure

```
soil-npk-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py                  ← Full training pipeline
│   └── app.py                    ← Streamlit UI
├── data/
│   └── lucas_npk_clean.csv       ← 20,471 samples (N, P, K, pH, Clay, OC...)
├── results/
│   ├── N_regression_plot.png
│   ├── K_regression_plot.png
│   ├── P_confusion_matrix.png
│   └── npk_fold_metrics.csv
├── demo/
│   ├── sample_multispectral.tif
│   ├── ui_screenshot.png
│   └── demo_video.mp4
└── docs/
    └── NPK_Prediction_Presentation.pptx
```

---

## Team

| Name | Roll No |
|---|---|
| Ujjwal Kumar | 230104028 |
| Atul Yadav | 230102041 |

**Supervisor:** Dr. D. Neelamegam (Assistant Professor, ECE, IIIT Manipur)  
**Course:** EC3201 — Minor Project, Semester 6  
**Institute:** IIIT Senapati, Manipur | April 2026

---

## References

1. Barnes & Dhanoa (1989) — SNV Normalization
2. Savitzky & Golay (1964) — SG Filtering
3. Wold & Sjöström (2001) — PLS Regression
4. Chen & Guestrin (2016) — XGBoost
5. Lucas et al. (2015) — LUCAS Soil Dataset
6. Vullaganti et al. (2023) — AI-Augmented Hyperspectral Modeling
