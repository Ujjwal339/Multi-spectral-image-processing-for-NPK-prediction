import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tifffile as tiff
import io

st.set_page_config(page_title="Multispectral NPK AI", layout="wide")
st.title("🛰️ Multispectral Soil NPK Predictor")
st.markdown("Upload a Multispectral Image (.tif) or Spectral CSV to predict soil chemistry using Hierarchical Data Fusion.")

# --- 1. LOAD THE MASTER PIPELINE ---
@st.cache_resource
def load_pipeline():
    return joblib.load('FINAL_NPK_PIPELINE.pkl')

try:
    pipeline = load_pipeline()
    st.sidebar.success("✅ AI Pipeline Loaded Successfully")
except Exception as e:
    st.error(f"Could not load the pipeline. Error: {e}")
    st.stop()

# --- 2. USER INTERFACE (SIDEBAR) ---
st.sidebar.header("🔬 Input Soil Properties")
st.sidebar.markdown("Provide the known physical properties for this sample location:")
ph = st.sidebar.slider("pH Level", 4.0, 9.0, 6.5, 0.1)
clay = st.sidebar.slider("Clay (%)", 0.0, 100.0, 20.0, 1.0)
silt = st.sidebar.slider("Silt (%)", 0.0, 100.0, 40.0, 1.0)
sand = st.sidebar.slider("Sand (%)", 0.0, 100.0, 40.0, 1.0)
oc = st.sidebar.slider("Organic Carbon (g/kg)", 0.0, 500.0, 25.0, 1.0)

# --- 3. MULTISPECTRAL FILE UPLOAD ---
st.markdown("### 📤 Upload Multispectral Data")
uploaded_file = st.file_uploader("Upload a Multispectral .tif image or extracted .csv (like 01.csv)", type=["tif", "tiff", "csv"])

if uploaded_file is not None:
    
    with st.spinner("Extracting Multispectral Signatures..."):
        try:
            # SCENARIO A: User uploaded a Multispectral Image (.tif)
            if uploaded_file.name.endswith(('.tif', '.tiff')):
                img = tiff.imread(uploaded_file)
                # Assuming shape is (Bands, Height, Width) or (Height, Width, Bands)
                if img.ndim == 3:
                    # Calculate the average reflectance for each band across the whole image
                    if img.shape[0] < img.shape[2]: # (Bands, H, W)
                        multi_signature = np.mean(img, axis=(1, 2))
                    else: # (H, W, Bands)
                        multi_signature = np.mean(img, axis=(0, 1))
                else:
                    st.error("Invalid image shape. Expected a multi-band TIFF.")
                    st.stop()
                
                st.success(f"Extracted average reflectance across {len(multi_signature)} bands from the image.")

            # SCENARIO B: User uploaded the extracted CSV (like 01.csv)
            elif uploaded_file.name.endswith('.csv'):
                # Read the CSV (skipping headers if it's formatted like 01.csv)
                df_spec = pd.read_csv(uploaded_file, skiprows=4) # Skips the metadata lines
                # Extract the wavelength column and the first sample column
                wavelengths_multi = df_spec.iloc[:, 0].values
                multi_signature = df_spec.iloc[:, 1].values # Taking the first pixel/sample
                st.success(f"Extracted {len(multi_signature)} spectral bands from CSV.")

            # --- THE MAGIC TRICK: INTERPOLATION ---
            # We must stretch the user's N bands to the 4,200 bands the LUCAS AI expects.
            # (Assuming LUCAS bands range roughly from 400nm to 2500nm uniformly)
            lucas_expected_features = pipeline['scaler_spec'].n_features_in_
            
            # Create a dummy wavelength array for the multispectral input and the LUCAS target
            dummy_multi_waves = np.linspace(400, 1000, len(multi_signature)) 
            lucas_waves = np.linspace(400, 2500, lucas_expected_features)
            
            # Interpolate the multispectral signature to fit 4200 bands
            aligned_spectrum = np.interp(lucas_waves, dummy_multi_waves, multi_signature)
            
            # Reshape for the ML model (1 sample, 4200 bands)
            final_spectrum_array = aligned_spectrum.reshape(1, -1)
            
        except Exception as e:
            st.error(f"Error processing the multispectral file: {e}")
            st.stop()

    # --- 4. PREDICTION ENGINE ---
    if st.button("🚀 Analyze Multispectral Signature", type="primary"):
        with st.spinner("Applying Hierarchical Data Fusion..."):
            
            # Base Chemistry Engine
            clay_ph = clay * ph
            clay_squared = clay ** 2
            ph_squared = ph ** 2
            cec_proxy = (clay * 0.5) + (oc * 2.0)
            
            base_inputs = np.array([[ph, clay, silt, sand, oc, clay_ph, clay_squared, ph_squared, cec_proxy]])
            
            X_chem_N = pipeline['imputer_N'].transform(base_inputs)
            X_chem_P = pipeline['imputer_P'].transform(base_inputs)
            
            # Scale the aligned spectrum
            X_spec_scaled = pipeline['scaler_spec'].transform(final_spectrum_array)
            
            # N Model
            X_spec_N = pipeline['pls_N'].transform(X_spec_scaled)
            pred_N = pipeline['model_N'].predict(np.hstack((X_chem_N, X_spec_N)))[0]
            
            # P Model
            X_spec_P = pipeline['pls_P'].transform(X_spec_scaled)
            pred_P = np.expm1(pipeline['model_P'].predict(np.hstack((X_chem_P, X_spec_P)))[0])
            
            # K Model (Uses N and P)
            oc_n = oc * pred_N
            n_p_ratio = pred_N / (pred_P + 1e-10)
            k_inputs = np.array([[ph, clay, silt, sand, oc, pred_N, pred_P, clay_ph, oc_n, clay_squared, ph_squared, cec_proxy, n_p_ratio]])
            X_chem_K = pipeline['imputer_K'].transform(k_inputs)
            
            X_spec_K = pipeline['pls_K'].transform(X_spec_scaled)
            pred_K = np.expm1(pipeline['model_K'].predict(np.hstack((X_chem_K, X_spec_K)))[0])
            
        # Display Results
        st.markdown("### 🌱 Multispectral Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Nitrogen (N)", f"{pred_N:.2f} g/kg")
        col2.metric("Phosphorus (P)", f"{pred_P:.2f} mg/kg")
        col3.metric("Potassium (K)", f"{pred_K:.2f} mg/kg")
        
        with st.expander("View Interpolated Spectral Signature"):
            st.line_chart(pd.DataFrame(aligned_spectrum, columns=["Reflectance"]))
