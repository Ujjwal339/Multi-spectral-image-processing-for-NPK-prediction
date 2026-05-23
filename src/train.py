import os
import glob
import time
import gc
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

START_TIME = time.time()

SG_WINDOW = 21
SG_POLY = 2
SPECTRAL_STACKS = 2

PLS_N_COMP = 20
PLS_P_COMP = 25
PLS_K_COMP = 25

P_BOUNDS = [1.4, 12.6, 24.9, 44.4]

P_CLASSES = [
    "Low (1.4-12.6)",
    "Med-Low (12.6-24.9)",
    "Med-High (24.9-44.4)",
    "High (44.4+)"
]

print("Loading configuration...")

spectral_dir = "/home/ujjwal/ssd/Soil_NPK/LUCAS2015_spectra/LUCAS2015_Soil_Spectra_EU28"
files = glob.glob(os.path.join(spectral_dir, "*.csv"))

if len(files) == 0:
    raise ValueError("No spectral CSV files found.")

spectral_raw = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)

metadata_cols = [
    c for c in spectral_raw.columns
    if not c.replace('.', '').isdigit()
]

band_cols = [
    c for c in spectral_raw.columns
    if c not in metadata_cols
]

lucas_master = pd.read_csv("lucas_npk_clean.csv")

print(f"Spectral data shape: {spectral_raw.shape}")
print(f"Master data shape: {lucas_master.shape}")
print(f"Spectral bands: {len(band_cols)}")

# Chemistry-based engineered features
lucas_master["Clay_pH"] = lucas_master["Clay"] * lucas_master["pH"]

lucas_master["CEC_proxy"] = (
    lucas_master["Clay"] * 0.5 +
    lucas_master["OC"] * 2.0
)

lucas_master["Texture_Index"] = (
    lucas_master["Clay"] +
    lucas_master["Silt"]
) / 2

lucas_master["P_Buffer_Index"] = (
    lucas_master["Clay"] *
    lucas_master["pH"]
)

lucas_master["OC_Clay_Ratio"] = (
    lucas_master["OC"] /
    (lucas_master["Clay"] + 1)
)

lucas_master["Soil_Reactivity"] = (
    lucas_master["pH"] *
    lucas_master["OC"]
)

lucas_master["Clay_OC_Product"] = (
    lucas_master["Clay"] *
    lucas_master["OC"]
)

lucas_master["Texture_Reactivity"] = (
    lucas_master["Clay"] +
    lucas_master["Silt"]
)

# Potassium-specific proxy features
lucas_master["K_Exchangeable_Proxy"] = (
    lucas_master["Clay"] *
    lucas_master["pH"] * 0.1
)

lucas_master["K_Weathering_Index"] = (
    lucas_master["Sand"] /
    (lucas_master["Clay"] + 1)
)

features_base = [
    "pH",
    "Clay",
    "Silt",
    "Sand",
    "OC",
    "Clay_pH",
    "CEC_proxy",
    "Texture_Index",
    "P_Buffer_Index",
    "OC_Clay_Ratio",
    "Soil_Reactivity",
    "Clay_OC_Product",
    "Texture_Reactivity",
    "K_Exchangeable_Proxy",
    "K_Weathering_Index"
]

print(f"Engineered features: {len(features_base)}")

fused_data = pd.merge(
    lucas_master,
    spectral_raw,
    on="PointID",
    how="inner"
)

fused_data = fused_data.dropna(subset=["N", "P", "K"])

print(f"Fused dataset shape: {fused_data.shape}")

possible_group_cols = [
    "NUTS_0",
    "NUTS0",
    "COUNTRY",
    "country"
]

group_col = next(
    (col for col in possible_group_cols if col in fused_data.columns),
    None
)

if group_col is None:
    raise KeyError("No spatial group column found.")

groups = fused_data[group_col].values
unique_groups = np.unique(groups)

print(f"Geographic groups: {len(unique_groups)}")

country_dummies = pd.get_dummies(
    fused_data[group_col],
    prefix='Geo',
    dtype=np.float32
)

country_cols = country_dummies.columns.tolist()

fused_data = pd.concat(
    [fused_data, country_dummies],
    axis=1
)

features_combined = features_base + country_cols

print(f"Total features: {len(features_combined)}")

y_N = np.log1p(fused_data["N"].values)
y_K = np.log1p(fused_data["K"].values)
y_P_raw = fused_data["P"].values


def classify_P(p_values, bounds):
    classes = np.zeros(len(p_values), dtype=int)

    for i in range(len(bounds) - 1):
        mask = (
            (p_values >= bounds[i]) &
            (p_values < bounds[i + 1])
        )
        classes[mask] = i

    classes[p_values >= bounds[-1]] = len(bounds) - 1

    return classes


y_P = classify_P(y_P_raw, P_BOUNDS)

unique_classes, counts = np.unique(y_P, return_counts=True)

print("\nPhosphorus class distribution:")

for cls, count in zip(unique_classes, counts):
    print(f"{P_CLASSES[cls]}: {count}")

X_chem = fused_data[features_combined].values
raw_band_data = fused_data[band_cols].values.astype(np.float32)

del spectral_raw
del lucas_master
del fused_data
del country_dummies

gc.collect()

# Spectral preprocessing: SNV normalization + SG filtering
mean_spec = np.mean(raw_band_data, axis=1, keepdims=True)

std_spec = np.maximum(
    np.std(raw_band_data, axis=1, keepdims=True),
    1e-6
)

snv_data = (raw_band_data - mean_spec) / std_spec

spec_smooth = savgol_filter(
    snv_data,
    window_length=SG_WINDOW,
    polyorder=SG_POLY,
    axis=1
).astype(np.float32)

spec_d1 = savgol_filter(
    snv_data,
    window_length=SG_WINDOW,
    polyorder=SG_POLY,
    deriv=1,
    axis=1
).astype(np.float32)

X_spec = np.hstack(
    (spec_smooth, spec_d1)
).astype(np.float32)

print(f"Spectral feature matrix: {X_spec.shape}")

del snv_data
del spec_smooth
del spec_d1
del mean_spec
del std_spec
del raw_band_data

gc.collect()

print("\nRunning spatial cross-validation...")

gkf = GroupKFold(n_splits=5)

imputer = SimpleImputer(strategy="median")
scaler_spec = StandardScaler()

results = []
fold = 1

all_true_N = []
all_pred_N = []

all_true_K = []
all_pred_K = []

all_true_P = []
all_pred_P = []

for train_idx, test_idx in gkf.split(X_spec, y_N, groups):

    print(f"\nFold {fold}/5")

    X_chem_train = imputer.fit_transform(X_chem[train_idx])
    X_chem_test = imputer.transform(X_chem[test_idx])

    X_spec_train = scaler_spec.fit_transform(X_spec[train_idx])
    X_spec_test = scaler_spec.transform(X_spec[test_idx])

    # Nitrogen model
    pls_N = PLSRegression(n_components=PLS_N_COMP)

    pls_N.fit(X_spec_train, y_N[train_idx])

    XN_train = np.hstack((
        X_chem_train,
        pls_N.transform(X_spec_train)
    ))

    XN_test = np.hstack((
        X_chem_test,
        pls_N.transform(X_spec_test)
    ))

    model_N = xgb.XGBRegressor(
        n_estimators=320,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    model_N.fit(XN_train, y_N[train_idx])

    pred_N = np.expm1(model_N.predict(XN_test))

    all_true_N.extend(np.expm1(y_N[test_idx]))
    all_pred_N.extend(pred_N)

    r2_N = r2_score(
        np.expm1(y_N[test_idx]),
        pred_N
    )

    rmse_N = np.sqrt(
        mean_squared_error(
            np.expm1(y_N[test_idx]),
            pred_N
        )
    )

    print(f"N -> R²: {r2_N:.3f}, RMSE: {rmse_N:.2f}")

    # Phosphorus model
    pls_P = PLSRegression(n_components=PLS_P_COMP)

    pls_P.fit(X_spec_train, y_P[train_idx])

    XP_train = np.hstack((
        X_chem_train,
        pls_P.transform(X_spec_train)
    ))

    XP_test = np.hstack((
        X_chem_test,
        pls_P.transform(X_spec_test)
    ))

    model_P = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=4,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    model_P.fit(XP_train, y_P[train_idx])

    pred_P = model_P.predict(XP_test)

    all_true_P.extend(y_P[test_idx])
    all_pred_P.extend(pred_P)

    acc_P = accuracy_score(
        y_P[test_idx],
        pred_P
    )

    f1_P = f1_score(
        y_P[test_idx],
        pred_P,
        average='macro',
        zero_division=0
    )

    print(f"P -> Accuracy: {100 * acc_P:.1f}%, F1: {f1_P:.3f}")

    # Potassium model
    pls_K = PLSRegression(n_components=PLS_K_COMP)

    pls_K.fit(X_spec_train, y_K[train_idx])

    XK_train = np.hstack((
        X_chem_train,
        pls_K.transform(X_spec_train)
    ))

    XK_test = np.hstack((
        X_chem_test,
        pls_K.transform(X_spec_test)
    ))

    model_K = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    model_K.fit(XK_train, y_K[train_idx])

    pred_K = np.expm1(model_K.predict(XK_test))

    all_true_K.extend(np.expm1(y_K[test_idx]))
    all_pred_K.extend(pred_K)

    r2_K = r2_score(
        np.expm1(y_K[test_idx]),
        pred_K
    )

    rmse_K = np.sqrt(
        mean_squared_error(
            np.expm1(y_K[test_idx]),
            pred_K
        )
    )

    print(f"K -> R²: {r2_K:.3f}, RMSE: {rmse_K:.2f}")

    results.append([
        fold,
        r2_N,
        acc_P,
        r2_K,
        rmse_N,
        f1_P,
        rmse_K
    ])

    fold += 1

    del X_chem_train
    del X_chem_test
    del X_spec_train
    del X_spec_test

    del pls_N
    del pls_P
    del pls_K

    del model_N
    del model_P
    del model_K

    gc.collect()

results_df = pd.DataFrame(
    results,
    columns=[
        "Fold",
        "N_R2",
        "P_Acc",
        "K_R2",
        "N_RMSE",
        "P_F1",
        "K_RMSE"
    ]
)

print("\nGenerating evaluation plots...")

true_N = np.array(all_true_N)
pred_N = np.array(all_pred_N)

true_K = np.array(all_true_K)
pred_K = np.array(all_pred_K)

true_P = np.array(all_true_P)
pred_P = np.array(all_pred_P)

# Nitrogen regression plot
plt.figure()

plt.scatter(true_N, pred_N, alpha=0.5)

plt.xlabel("Actual Nitrogen")
plt.ylabel("Predicted Nitrogen")

plt.title("Nitrogen Regression")

min_val = min(true_N.min(), pred_N.min())
max_val = max(true_N.max(), pred_N.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.savefig("N_regression_plot.png", dpi=300)

plt.close()

# Potassium regression plot
plt.figure()

plt.scatter(true_K, pred_K, alpha=0.5)

plt.xlabel("Actual Potassium")
plt.ylabel("Predicted Potassium")

plt.title("Potassium Regression")

min_val = min(true_K.min(), pred_K.min())
max_val = max(true_K.max(), pred_K.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.savefig("K_regression_plot.png", dpi=300)

plt.close()

# Phosphorus confusion matrix
cm = confusion_matrix(true_P, pred_P)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=P_CLASSES,
    yticklabels=P_CLASSES
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title("Phosphorus Confusion Matrix")

plt.tight_layout()
plt.savefig("P_confusion_matrix.png", dpi=300)

plt.close()

results_df.to_csv("npk_fold_metrics.csv", index=False)

print("\nCross-validation summary:")

print(
    f"N R²: "
    f"{results_df['N_R2'].mean():.3f} ± "
    f"{results_df['N_R2'].std():.3f}"
)

print(
    f"P Accuracy: "
    f"{100 * results_df['P_Acc'].mean():.1f}% ± "
    f"{100 * results_df['P_Acc'].std():.1f}%"
)

print(
    f"K R²: "
    f"{results_df['K_R2'].mean():.3f} ± "
    f"{results_df['K_R2'].std():.3f}"
)

print("\nTraining final models on full dataset...")

X_chem_full = imputer.fit_transform(X_chem)
X_spec_full = scaler_spec.fit_transform(X_spec)

# Final Nitrogen model
pls_N_full = PLSRegression(n_components=PLS_N_COMP)

pls_N_full.fit(X_spec_full, y_N)

XN_full = np.hstack((
    X_chem_full,
    pls_N_full.transform(X_spec_full)
))

model_N_full = xgb.XGBRegressor(
    n_estimators=320,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbose=0
)

model_N_full.fit(XN_full, y_N)

# Final Phosphorus model
pls_P_full = PLSRegression(n_components=PLS_P_COMP)

pls_P_full.fit(X_spec_full, y_P)

XP_full = np.hstack((
    X_chem_full,
    pls_P_full.transform(X_spec_full)
))

model_P_full = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=4,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

model_P_full.fit(XP_full, y_P)

# Final Potassium model
pls_K_full = PLSRegression(n_components=PLS_K_COMP)

pls_K_full.fit(X_spec_full, y_K)

XK_full = np.hstack((
    X_chem_full,
    pls_K_full.transform(X_spec_full)
))

model_K_full = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.7,
    reg_alpha=0.2,
    reg_lambda=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbose=0
)

model_K_full.fit(XK_full, y_K)

print("Saving production pipeline...")

pipeline = {
    "model_N": model_N_full,
    "model_P": model_P_full,
    "model_K": model_K_full,

    "pls_N": pls_N_full,
    "pls_P": pls_P_full,
    "pls_K": pls_K_full,

    "imputer": imputer,
    "scaler_spec": scaler_spec,

    "pls_components": {
        "N": PLS_N_COMP,
        "P": PLS_P_COMP,
        "K": PLS_K_COMP
    },

    "sg_window": SG_WINDOW,
    "sg_poly": SG_POLY,
    "spectral_stacks": SPECTRAL_STACKS,

    "features_base": features_base,
    "features_combined": features_combined,

    "num_base_features": len(features_base),
    "num_geo_features": len(country_cols),

    "p_bounds": P_BOUNDS,
    "p_classes": P_CLASSES,

    "has_derivatives": True,
    "preprocessing": "SNV + SG1",

    "k_features_added": [
        "K_Exchangeable_Proxy",
        "K_Weathering_Index"
    ]
}

joblib.dump(
    pipeline,
    "p_fixed.pkl",
    compress=9
)

elapsed = (time.time() - START_TIME) / 60.0

print("\nTraining complete.")
print(f"Total time: {elapsed:.1f} minutes")

print("\nSaved artifacts:")
print("- p_fixed.pkl")
print("- npk_fold_metrics.csv")
print("- N_regression_plot.png")
print("- K_regression_plot.png")
print("- P_confusion_matrix.png")