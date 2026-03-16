"""
THE TRUE PARALLEL NPK FUSION (NO DATA LEAKAGE)
Combines the best methodologies for N, P, and K into independent pipelines.
Ensures that models cannot "cheat" by seeing their own target variables.
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("  🌟 THE TRUE PARALLEL NPK FUSION (LEAKAGE-FREE)  ")
print("="*80)

# ============================================================================
# STEP 1: LOAD SPECTRA
# ============================================================================
print("\n[STEP 1] Loading LUCAS Spectral CSVs...")
spectral_dir = os.path.expanduser('~/Lucas/LUCAS2015_spectra/LUCAS2015_Soil_Spectra_EU28')
spectral_raw = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(spectral_dir, "spectra_*.csv"))], ignore_index=True)

metadata_cols = ['source', 'SampleID', 'NUTS_0', 'SampleN']
spectral_numeric = spectral_raw.drop(columns=[col for col in metadata_cols if col in spectral_raw.columns])
spectral_averaged = spectral_numeric.groupby('PointID').mean().reset_index()
band_cols = [col for col in spectral_averaged.columns if col != 'PointID']

# ============================================================================
# STEP 2: LOAD MASTER DATA & ENGINEER SAFE FEATURES
# ============================================================================
print("\n[STEP 2] Loading Master Data & Engineering Safe Features...")
lucas_master = pd.read_csv('lucas_npk_clean.csv')


lucas_master['Clay_pH'] = lucas_master['Clay'] * lucas_master['pH']
lucas_master['Clay_squared'] = lucas_master['Clay'] ** 2
lucas_master['pH_squared'] = lucas_master['pH'] ** 2
lucas_master['CEC_proxy'] = (lucas_master['Clay'] * 0.5) + (lucas_master['OC'] * 2.0)


lucas_master['OC_N'] = lucas_master['OC'] * lucas_master['N']
lucas_master['N_P_ratio'] = lucas_master['N'] / (lucas_master['P'] + 1e-10)


fused_data = pd.merge(lucas_master, spectral_averaged, on='PointID', how='inner')
fused_data = fused_data.dropna(subset=['N', 'P', 'K'])
print(f"  Total samples ready for Modeling: {len(fused_data)}")

# ============================================================================
# STEP 3: CREATE STRICT FEATURE SETS & TARGETS
# ============================================================================
# N-Model can only see physical properties and OC
features_N = ['pH', 'Clay', 'Silt', 'Sand', 'OC', 'Clay_pH', 'Clay_squared', 'pH_squared', 'CEC_proxy']


features_P = features_N.copy() 


features_K = ['pH', 'Clay', 'Silt', 'Sand', 'OC', 'N', 'P', 'Clay_pH', 'OC_N', 'Clay_squared', 'pH_squared', 'CEC_proxy', 'N_P_ratio']

X_chem_N = fused_data[features_N].values
X_chem_P = fused_data[features_P].values
X_chem_K = fused_data[features_K].values
X_spec = fused_data[band_cols].values


y_N = fused_data['N'].values
y_P_log = np.log1p(fused_data['P'].values)
y_K_log = np.log1p(fused_data['K'].values)

indices = np.arange(len(fused_data))
idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

# ============================================================================
# STEP 4: IMPUTE CHEMISTRY & SCALE SPECTRA
# ============================================================================
print("\n[STEP 4] Applying Model-Specific Imputation & Scaling...")

imputer_N = SimpleImputer(strategy='median')
X_chem_train_N = imputer_N.fit_transform(X_chem_N[idx_train])
X_chem_test_N = imputer_N.transform(X_chem_N[idx_test])

imputer_P = SimpleImputer(strategy='median')
X_chem_train_P = imputer_P.fit_transform(X_chem_P[idx_train])
X_chem_test_P = imputer_P.transform(X_chem_P[idx_test])

imputer_K = SimpleImputer(strategy='median')
X_chem_train_K = imputer_K.fit_transform(X_chem_K[idx_train])
X_chem_test_K = imputer_K.transform(X_chem_K[idx_test])

scaler_spec = StandardScaler()
X_spec_train_scaled = scaler_spec.fit_transform(X_spec[idx_train])
X_spec_test_scaled = scaler_spec.transform(X_spec[idx_test])

# ============================================================================
# STEP 5: PARALLEL TRAINING WITH XGBOOST
# ============================================================================
print("\n[STEP 5] Training Independent Target-Aware XGBoost Models...")

# ---------------------------------------------------------
# 1. NITROGEN (N) PIPELINE
# ---------------------------------------------------------
print("  -> Extracting N-Spectra & Training Nitrogen...")
pls_N = PLSRegression(n_components=15)
X_spec_train_N = pls_N.fit_transform(X_spec_train_scaled, y_N[idx_train])[0]
X_spec_test_N = pls_N.transform(X_spec_test_scaled)

X_train_N_final = np.hstack((X_chem_train_N, X_spec_train_N))
X_test_N_final = np.hstack((X_chem_test_N, X_spec_test_N))


model_N = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model_N.fit(X_train_N_final, y_N[idx_train])
pred_N_test = model_N.predict(X_test_N_final)

# ---------------------------------------------------------
# 2. PHOSPHORUS (P) PIPELINE
# ---------------------------------------------------------
print("  -> Extracting P-Spectra & Training Phosphorus...")
pls_P = PLSRegression(n_components=15)
X_spec_train_P = pls_P.fit_transform(X_spec_train_scaled, y_P_log[idx_train])[0]
X_spec_test_P = pls_P.transform(X_spec_test_scaled)

X_train_P_final = np.hstack((X_chem_train_P, X_spec_train_P))
X_test_P_final = np.hstack((X_chem_test_P, X_spec_test_P))

model_P = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    reg_lambda=2.0, 
    random_state=42,
    n_jobs=-1
)
model_P.fit(X_train_P_final, y_P_log[idx_train])
pred_P_test = np.expm1(model_P.predict(X_test_P_final))

# ---------------------------------------------------------
# 3. POTASSIUM (K) PIPELINE
# ---------------------------------------------------------
print("  -> Extracting K-Spectra & Training Potassium...")
pls_K = PLSRegression(n_components=15)
X_spec_train_K = pls_K.fit_transform(X_spec_train_scaled, y_K_log[idx_train])[0]
X_spec_test_K = pls_K.transform(X_spec_test_scaled)

X_train_K_final = np.hstack((X_chem_train_K, X_spec_train_K))
X_test_K_final = np.hstack((X_chem_test_K, X_spec_test_K))


model_K = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.8,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1
)
model_K.fit(X_train_K_final, y_K_log[idx_train])
pred_K_test = np.expm1(model_K.predict(X_test_K_final))

# ============================================================================
# STEP 6: EVALUATION & EXPORT
# ============================================================================
print("\n" + "="*80)
print("  🏆 FINAL RESULTS: TRUE PARALLEL FUSION")
print("="*80)

y_P_true = fused_data['P'].values[idx_test]
y_K_true = fused_data['K'].values[idx_test]

print(f"  Nitrogen (N) R²   : {r2_score(y_N[idx_test], pred_N_test):.3f}")
print(f"  Phosphorus (P) R² : {r2_score(y_P_true, pred_P_test):.3f}  |  RMSE: {np.sqrt(mean_squared_error(y_P_true, pred_P_test)):.2f} mg/kg")
print(f"  Potassium (K) R²  : {r2_score(y_K_true, pred_K_test):.3f}  |  RMSE: {np.sqrt(mean_squared_error(y_K_true, pred_K_test)):.2f} mg/kg")
print("="*80)


joblib.dump({
    'imputer_N': imputer_N, 'imputer_P': imputer_P, 'imputer_K': imputer_K,
    'scaler_spec': scaler_spec,
    'pls_N': pls_N, 'pls_P': pls_P, 'pls_K': pls_K,
    'model_N': model_N, 'model_P': model_P, 'model_K': model_K,
    'features_N': features_N, 'features_P': features_P, 'features_K': features_K
}, 'FINAL_NPK_PIPELINE.pkl')
print("\n[✓] Final Pipeline saved as 'FINAL_NPK_PIPELINE.pkl'")
