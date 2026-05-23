import gc
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import requests
import streamlit as st

from pykrige.ok import OrdinaryKriging
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from scipy.signal import savgol_filter

st.set_page_config(
    page_title="Spatial NPK Mapper",
    layout="wide"
)

st.title("Spatial Soil NPK Mapping Engine")

st.markdown(
    "Upload multispectral GeoTIFF data to predict "
    "soil nutrients and visualize spatial nutrient maps."
)

@st.cache_resource
def load_pipeline(path="p_fixed.pkl"):
    return joblib.load(path)

try:
    pipeline = load_pipeline()

except Exception as e:
    st.error(f"Pipeline loading failed: {e}")
    st.info("Ensure 'p_fixed.pkl' exists in the same directory.")
    st.stop()


@st.cache_data
def fetch_soilgrids_data(lat, lon):

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    params = {
        "lon": lon,
        "lat": lat,
        "property": ["phh2o", "clay", "silt", "sand", "soc"],
        "depth": "0-5cm",
        "value": "mean"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        ).json()

        layers = {
            l['name']: l['depths'][0]['values']['mean']
            for l in response['properties']['layers']
        }

        return (
            layers.get('phh2o') / 10,
            layers.get('clay') / 10,
            layers.get('silt') / 10,
            layers.get('sand') / 10,
            layers.get('soc') / 10
        )

    except Exception:
        return None


def get_country_code(lat, lon):

    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=json&lat={lat}&lon={lon}&zoom=3"
        )

        response = requests.get(
            url,
            headers={'User-Agent': 'GeoAware-NPK-Mapper'},
            timeout=10
        ).json()

        country_code = response.get(
            'address',
            {}
        ).get(
            'country_code',
            ''
        ).upper()

        if country_code == 'GB':
            return 'UK'

        if country_code == 'GR':
            return 'EL'

        return country_code

    except Exception:
        return None


def normalize(arr):

    return (
        (arr - np.min(arr)) /
        ((np.max(arr) - np.min(arr)) + 1e-10)
    )


def predict_npk(X_spec_raw, X_chem, pipeline):

    sg_window = pipeline.get("sg_window", 21)
    sg_poly = pipeline.get("sg_poly", 2)

    spec_smooth = savgol_filter(
        X_spec_raw,
        window_length=sg_window,
        polyorder=sg_poly,
        axis=1
    )

    spec_d1 = savgol_filter(
        X_spec_raw,
        window_length=sg_window,
        polyorder=sg_poly,
        deriv=1,
        axis=1
    )

    X_spec_scaled = pipeline['scaler_spec'].transform(
        np.hstack([spec_smooth, spec_d1]).astype(np.float32)
    )

    XN = np.hstack((
        X_chem,
        pipeline['pls_N'].transform(X_spec_scaled)
    ))

    pred_N = np.expm1(
        pipeline['model_N'].predict(XN)
    )

    XP = np.hstack((
        X_chem,
        pipeline['pls_P'].transform(X_spec_scaled)
    ))

    pred_P_class = pipeline['model_P'].predict(XP)

    XK = np.hstack((
        X_chem,
        pipeline['pls_K'].transform(X_spec_scaled)
    ))

    pred_K = np.expm1(
        pipeline['model_K'].predict(XK)
    )

    return pred_N, pred_P_class, pred_K


st.markdown("### Upload Data")

uploaded_file = st.file_uploader(
    "Upload Multispectral GeoTIFF",
    type=["tif", "tiff", "csv"]
)

if not uploaded_file:
    st.info("Awaiting file upload...")
    st.stop()


# CSV workflow
if uploaded_file.name.endswith(".csv"):

    df = pd.read_csv(uploaded_file)

    st.success(f"CSV loaded: {len(df)} rows")

    if (
        "Latitude" not in df.columns or
        "Longitude" not in df.columns
    ):
        st.error(
            "CSV must contain 'Latitude' and 'Longitude' columns."
        )
        st.stop()

    lat = df["Latitude"].mean()
    lon = df["Longitude"].mean()

    band_cols = [
        c for c in df.columns
        if c not in ["Latitude", "Longitude", "PointID"]
        and not c.startswith("Unnamed")
    ]

    samples = df[band_cols].values.astype(np.float32)

    soil = fetch_soilgrids_data(lat, lon)

    if soil is None:
        soil = (6.5, 20.0, 40.0, 40.0, 25.0)

    ph, clay, silt, sand, oc = soil

    country_code = get_country_code(lat, lon)

    base_chem = np.tile([
        ph,
        clay,
        silt,
        sand,
        oc,
        clay * ph,
        clay * 0.5 + oc * 2.0,
        (clay + silt) / 2,
        clay * ph,
        oc / (clay + 1),
        ph * oc,
        clay * oc,
        clay + silt,
        clay * ph * 0.1,
        sand / (clay + 1)
    ], (len(samples), 1))

    geo_features = [
        f for f in pipeline['features_combined']
        if f.startswith("Geo_")
    ]

    geo_array = np.zeros(
        (len(samples), len(geo_features)),
        dtype=np.float32
    )

    if (
        country_code and
        f"Geo_{country_code}" in geo_features
    ):
        geo_array[
            :,
            geo_features.index(f"Geo_{country_code}")
        ] = 1.0

    X_chem_final = pipeline['imputer'].transform(
        np.hstack([base_chem, geo_array])
    )

    lucas_bands = (
        pipeline['scaler_spec'].n_features_in_ //
        pipeline.get('spectral_stacks', 2)
    )

    interp = interp1d(
        np.linspace(400, 1000, len(band_cols)),
        samples,
        axis=1,
        fill_value="extrapolate"
    )

    stretched = np.nan_to_num(
        interp(np.linspace(400, 2500, lucas_bands))
    )

    with st.spinner("Running inference..."):

        pred_N, pred_P_class, pred_K = predict_npk(
            stretched,
            X_chem_final,
            pipeline
        )

        df["Predicted_N"] = pred_N
        df["Predicted_P_Class"] = pred_P_class
        df["Predicted_K"] = pred_K

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Avg Nitrogen (N)",
        f"{np.mean(pred_N):.2f} g/kg"
    )

    col2.metric(
        "Avg Phosphorus Zone",
        pipeline['p_classes'][int(np.median(pred_P_class))]
    )

    col3.metric(
        "Avg Potassium (K)",
        f"{np.mean(pred_K):.2f} mg/kg"
    )

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Predicted_N",
        color_continuous_scale="Greens",
        size_max=15,
        zoom=14,
        mapbox_style="carto-positron",
        title="Nitrogen Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    csv_data = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Predictions CSV",
        data=csv_data,
        file_name="NPK_Predictions.csv",
        mime="text/csv"
    )

    st.stop()


# GeoTIFF workflow
with st.spinner("Loading GeoTIFF and syncing SoilGrids..."):

    with MemoryFile(uploaded_file) as memfile:

        with memfile.open() as src:

            if src.crs is None:

                st.warning(
                    "GeoTIFF missing GPS metadata. "
                    "Using fallback coordinates."
                )

                lat, lon = 50.0, 10.0

            else:
                bounds = src.bounds
                img_crs = src.crs

                cx = (bounds.left + bounds.right) / 2
                cy = (bounds.bottom + bounds.top) / 2

                lon_arr, lat_arr = transform(
                    src.crs,
                    'epsg:4326',
                    [cx],
                    [cy]
                )

                lon = lon_arr[0]
                lat = lat_arr[0]

            st.info(
                f"Location: Lat {lat:.4f}, Lon {lon:.4f}"
            )

            scale = min(
                1.0,
                1500 / max(src.width, src.height)
            )

            new_h = int(src.height * scale)
            new_w = int(src.width * scale)

            img = src.read(
                out_shape=(
                    src.count,
                    new_h,
                    new_w
                ),
                resampling=rasterio.enums.Resampling.bilinear
            )

            img = np.moveaxis(
                img,
                0,
                -1
            ).astype(np.float32)

preview = normalize(img[:, :, :3])

ndvi = None

if img.shape[-1] >= 4:

    nir = img[:, :, 3]
    red = img[:, :, 2]

    ndvi = (
        (nir - red) /
        (nir + red + 1e-10)
    )

soil = fetch_soilgrids_data(lat, lon)

if soil is None:
    soil = (6.5, 20.0, 40.0, 40.0, 25.0)

ph, clay, silt, sand, oc = soil

country_code = get_country_code(lat, lon)

grid_size = 60

xv, yv = np.meshgrid(
    np.linspace(0, new_w - 1, grid_size, dtype=int),
    np.linspace(0, new_h - 1, grid_size, dtype=int)
)

samples = np.nan_to_num(
    img[yv.flatten(), xv.flatten(), :]
)

base_chem = np.tile([
    ph,
    clay,
    silt,
    sand,
    oc,
    clay * ph,
    clay * 0.5 + oc * 2.0,
    (clay + silt) / 2,
    clay * ph,
    oc / (clay + 1),
    ph * oc,
    clay * oc,
    clay + silt,
    clay * ph * 0.1,
    sand / (clay + 1)
], (len(samples), 1))

geo_features = [
    f for f in pipeline['features_combined']
    if f.startswith("Geo_")
]

geo_array = np.zeros(
    (len(samples), len(geo_features)),
    dtype=np.float32
)

if (
    country_code and
    f"Geo_{country_code}" in geo_features
):
    geo_array[
        :,
        geo_features.index(f"Geo_{country_code}")
    ] = 1.0

X_chem_final = pipeline['imputer'].transform(
    np.hstack([base_chem, geo_array])
)

lucas_bands = (
    pipeline['scaler_spec'].n_features_in_ //
    pipeline.get('spectral_stacks', 2)
)

interp = interp1d(
    np.linspace(400, 1000, img.shape[-1]),
    samples,
    axis=1,
    fill_value="extrapolate"
)

stretched = np.nan_to_num(
    interp(np.linspace(400, 2500, lucas_bands))
)

with st.spinner("Running ML inference..."):

    pred_N, pred_P_class, pred_K = predict_npk(
        stretched,
        X_chem_final,
        pipeline
    )


def get_kriging(values):

    values = np.nan_to_num(values)

    if (np.max(values) - np.min(values)) < 1e-8:
        return np.full(
            (grid_size, grid_size),
            values[0]
        )

    ok = OrdinaryKriging(
        xv.flatten(),
        yv.flatten(),
        values,
        variogram_model='spherical'
    )

    z, _ = ok.execute(
        'grid',
        np.linspace(0, new_w, grid_size),
        np.linspace(0, new_h, grid_size)
    )

    return z


grid_N = get_kriging(pred_N)
grid_P = get_kriging(pred_P_class)
grid_K = get_kriging(pred_K)

st.markdown("### Field Average Nutrients")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Nitrogen (N)",
    f"{np.mean(pred_N):.2f} g/kg"
)

c2.metric(
    "Phosphorus Zone (P)",
    f"{pipeline['p_classes'][int(np.median(pred_P_class))]}"
)

c3.metric(
    "Potassium (K)",
    f"{np.mean(pred_K):.2f} mg/kg"
)

zoom_y = new_h / grid_size
zoom_x = new_w / grid_size

gridN_full = zoom(grid_N, (zoom_y, zoom_x), order=1)
gridP_full = zoom(grid_P, (zoom_y, zoom_x), order=0)
gridK_full = zoom(grid_K, (zoom_y, zoom_x), order=1)

st.markdown("### Interactive Maps")

col_img1, col_img2 = st.columns(2)

with col_img1:

    fig_ortho = px.imshow(
        preview,
        title="Orthomosaic"
    )

    fig_ortho.update_traces(
        customdata=np.dstack((
            gridN_full,
            gridP_full,
            gridK_full
        )),
        hovertemplate=(
            "Nitrogen: %{customdata[0]:.2f} g/kg<br>"
            "P-Zone ID: %{customdata[1]:.0f}<br>"
            "Potassium: %{customdata[2]:.2f} mg/kg"
            "<extra></extra>"
        )
    )

    fig_ortho.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=700
    )

    fig_ortho.update_xaxes(visible=False)
    fig_ortho.update_yaxes(visible=False)

    st.plotly_chart(
        fig_ortho,
        use_container_width=True
    )

with col_img2:

    if ndvi is not None:

        fig_ndvi = px.imshow(
            ndvi,
            color_continuous_scale="RdYlGn",
            title="NDVI Index"
        )

        fig_ndvi.update_traces(
            customdata=np.dstack((
                gridN_full,
                gridP_full,
                gridK_full
            )),
            hovertemplate=(
                "NDVI: %{z:.2f}<br>"
                "Nitrogen: %{customdata[0]:.2f} g/kg<br>"
                "P-Zone ID: %{customdata[1]:.0f}<br>"
                "Potassium: %{customdata[2]:.2f} mg/kg"
                "<extra></extra>"
            )
        )

        fig_ndvi.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=700
        )

        fig_ndvi.update_xaxes(visible=False)
        fig_ndvi.update_yaxes(visible=False)

        st.plotly_chart(
            fig_ndvi,
            use_container_width=True
        )

    else:
        st.info(
            "NDVI unavailable. "
            "Requires a 4-band GeoTIFF with NIR."
        )

st.markdown("### Precision Heatmaps")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:

    fig_n = px.imshow(
        grid_N,
        color_continuous_scale="Greens",
        title="Nitrogen (g/kg)"
    )

    fig_n.update_xaxes(visible=False)
    fig_n.update_yaxes(visible=False)

    st.plotly_chart(
        fig_n,
        use_container_width=True
    )

with col_m2:

    fig_p = px.imshow(
        grid_P,
        color_continuous_scale="Reds",
        title="Phosphorus Zone"
    )

    fig_p.update_xaxes(visible=False)
    fig_p.update_yaxes(visible=False)

    st.plotly_chart(
        fig_p,
        use_container_width=True
    )

with col_m3:

    fig_k = px.imshow(
        grid_K,
        color_continuous_scale="Blues",
        title="Potassium (mg/kg)"
    )

    fig_k.update_xaxes(visible=False)
    fig_k.update_yaxes(visible=False)

    st.plotly_chart(
        fig_k,
        use_container_width=True
    )

st.markdown("### Combined NPK Dominance Map")

rgb_map = np.dstack((
    normalize(grid_P),
    normalize(grid_N),
    normalize(grid_K)
))

fig_rgb = px.imshow(
    rgb_map,
    title=(
        "Red = Phosphorus | "
        "Green = Nitrogen | "
        "Blue = Potassium"
    )
)

fig_rgb.update_xaxes(visible=False)
fig_rgb.update_yaxes(visible=False)

fig_rgb.update_layout(height=800)

st.plotly_chart(
    fig_rgb,
    use_container_width=True
)

st.markdown("### Export GIS Layers")

if 'bounds' in locals() and 'img_crs' in locals():

    new_transform = from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        grid_size,
        grid_size
    )

    def get_tiff_bytes(grid_data, is_rgb=False):

        with MemoryFile() as memfile:

            with memfile.open(
                driver='GTiff',
                height=grid_size,
                width=grid_size,
                count=3 if is_rgb else 1,
                dtype=(
                    rasterio.uint8
                    if is_rgb
                    else rasterio.float32
                ),
                crs=img_crs,
                transform=new_transform
            ) as dataset:

                if is_rgb:

                    dataset.write(
                        np.moveaxis(
                            (
                                grid_data * 255
                            ).astype(rasterio.uint8),
                            -1,
                            0
                        )
                    )

                else:

                    dataset.write(
                        grid_data.astype(rasterio.float32),
                        1
                    )

            return memfile.read()

    b_N = get_tiff_bytes(grid_N)
    b_P = get_tiff_bytes(grid_P)
    b_K = get_tiff_bytes(grid_K)

    b_RGB = get_tiff_bytes(
        rgb_map,
        is_rgb=True
    )

    dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)

    dl_c1.download_button(
        "Download N (.tif)",
        data=b_N,
        file_name="Nitrogen.tif",
        mime="image/tiff"
    )

    dl_c2.download_button(
        "Download P (.tif)",
        data=b_P,
        file_name="Phosphorus.tif",
        mime="image/tiff"
    )

    dl_c3.download_button(
        "Download K (.tif)",
        data=b_K,
        file_name="Potassium.tif",
        mime="image/tiff"
    )

    dl_c4.download_button(
        "Download RGB (.tif)",
        data=b_RGB,
        file_name="NPK_Dominance.tif",
        mime="image/tiff"
    )