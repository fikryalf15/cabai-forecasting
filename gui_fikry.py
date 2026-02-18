#!/usr/bin/env python3
"""
PREDIKSI HARGA KOMODITAS CABAI JAWA TIMUR DENGAN SVR
=====================================================
Complete Streamlit Application
Sidebar: Simplified - hanya konfigurasi model
"""

# ===============================
# IMPORT
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
import scipy
import pygad
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ===============================
# CUSTOM CSS
# ===============================
def add_custom_css():
    """Custom CSS styling"""
    st.markdown("""
    <style>
    /* Main styling */
    h1 {
        color: #2C3E50;
        font-size: 2.2em;
        margin-bottom: 5px;
    }
    
    h2 {
        color: #2C3E50;
        font-size: 1.6em;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #2C3E50;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        font-size: 1em;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 8px;
        padding: 20px;
        background-color: rgba(102, 126, 234, 0.05);
    }
    
    /* Info/Success/Warning boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: #e3f2fd !important;
        border-left: 5px solid #2196F3 !important;
    }
    
    .stSuccess {
        background-color: #e8f5e9 !important;
        border-left: 5px solid #4CAF50 !important;
    }
    
    .stWarning {
        background-color: #fff3e0 !important;
        border-left: 5px solid #FF9800 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Section styling */
    .sidebar-section {
        margin: 20px 0;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .sidebar-title {
        font-weight: bold;
        color: #667eea;
        font-size: 1.1em;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ===============================
# UTILITY FUNCTIONS
# ===============================
def create_sample_data():
    """Create sample data - Harga dalam ribuan rupiah (tanpa desimal)"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=365, freq='D')
    
    # Generate harga dalam range realistis (Rp 30.000 - Rp 80.000)
    base_rawit = 50000
    base_besar = 40000
    base_keriting = 35000
    
    # Variasi harga dengan trend dan seasonality
    cabai_rawit = base_rawit + np.cumsum(np.random.randn(365) * 300) + 5000 * np.sin(np.arange(365) * 2 * np.pi / 365)
    cabai_merah_besar = base_besar + np.cumsum(np.random.randn(365) * 250) + 4000 * np.sin(np.arange(365) * 2 * np.pi / 365)
    cabai_merah_keriting = base_keriting + np.cumsum(np.random.randn(365) * 200) + 3500 * np.sin(np.arange(365) * 2 * np.pi / 365)
    
    df = pd.DataFrame({
        'Tanggal': dates,
        'Cabai Rawit': np.maximum(cabai_rawit, 25000).astype(int),
        'Cabai Merah Besar': np.maximum(cabai_merah_besar, 20000).astype(int),
        'Cabai Merah Keriting': np.maximum(cabai_merah_keriting, 18000).astype(int)
    })
    
    return df

def plot_time_series_interactive(data, cabai):
    """Plot interaktif dengan plotly"""
    if cabai not in data.columns:
        raise ValueError(f"Kolom '{cabai}' tidak ditemukan di data")
    
    # Drop NA sementara untuk plotting
    ts = data[['Tanggal', cabai]].dropna()
    
    if ts.shape[0] == 0:
        raise ValueError(f"Data untuk '{cabai}' kosong atau semua NA, tidak dapat divisualisasikan")
    
    color_dict = {
        "Cabai Merah Besar": "#4ECDC4",
        "Cabai Rawit": "#FF6B6B",
        "Cabai Merah Keriting": "#1A535C"
    }
    
    color = color_dict.get(cabai, "#667eea")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Tanggal'],
        y=data[cabai],
        mode='lines+markers',
        name=cabai,
        line=dict(color=color, width=3),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor=color,
        opacity=0.3,
        hovertemplate='<b>Tanggal:</b> %{x|%d-%m-%Y}<br><b>Harga:</b> Rp %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Pola Deret Waktu {cabai}',
        xaxis_title='Waktu',
        yaxis_title='Harga (Rp)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        plot_bgcolor='rgba(240,240,240,0.5)',
    )
    
    return fig

# ===============================
# SET PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Harga Cabai - SVR",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

add_custom_css()

# ===============================
# SESSION STATE
# ===============================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'run_model' not in st.session_state:
    st.session_state.run_model = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ===============================
# TITLE
# ===============================
col1, col2 = st.columns([0.95, 0.05])
with col1:
    st.markdown("## Sistem Analisis Prediksi Harga Komoditas Cabai")

def train_svr_model(X_train, X_test, y_train, y_test, model_type):
    """Train SVR or SVR-GA model"""
    start_time = time.time()
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Train SVR
    if model_type == "SVR-GA (Genetic Algorithm)" :  # GA
        # For demo, using SVR dengan optimized params
        # =========================================================
        # LOG CONTAINER
        # =========================================================
        ga_log = []

        # =========================================================
        # FITNESS FUNCTION
        # =========================================================
        def fitness_func(ga_instance, solution, solution_idx):
            
            C, gamma, epsilon = solution
        
            model = SVR(
                kernel="rbf",
                C=C,
                gamma=gamma,
                epsilon=epsilon
            )

            tscv = TimeSeriesSplit(n_splits=5)
            rmses = []

            for train_idx, val_idx in tscv.split(X_train_scaled):

                X_tr = X_train_scaled[train_idx]
                X_val = X_train_scaled[val_idx]

                y_tr = y_train_scaled[train_idx]
                y_val = y_train_scaled[val_idx]

                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)

                # inverse ke harga asli
                y_val_inv = scaler_y.inverse_transform(
                    y_val.reshape(-1,1)
                )

                pred_inv = scaler_y.inverse_transform(
                    pred.reshape(-1,1)
                )

                rmse = np.sqrt(mean_squared_error(y_val_inv, pred_inv))
                rmses.append(rmse)

            return -np.mean(rmses)

        # =========================================================
        # LOG PER GENERASI
        # =========================================================
        def on_generation(ga_instance):

            best_solution, best_fitness, _ = ga_instance.best_solution()

            ga_log.append({
                "generation": ga_instance.generations_completed,
                "C": best_solution[0],
                "gamma": best_solution[1],
                "epsilon": best_solution[2],
                "RMSE": -best_fitness
            })

            print(
                f"GA | iter = {ga_instance.generations_completed:02d} | "
                f"Best RMSE = {-best_fitness:.3f}"
            )


        # =========================================================
        # SETTING GA
        # =========================================================
        ga = pygad.GA(
            num_generations=30,
            sol_per_pop=20,
            num_parents_mating=10,
            num_genes=3,
            gene_type=float,
            gene_space=[
                {"low": 0.1, "high": 100},
                {"low": 0.001, "high": 1},
                {"low": 0.001, "high": 0.5}
            ],
            fitness_func=fitness_func,
            on_generation=on_generation,
            mutation_percent_genes=20
        )

        # =========================================================
        # RUN GA
        # =========================================================
        ga.run()

        # =========================================================
        # HASIL LOG GA
        # =========================================================
        df_ga_results = pd.DataFrame(ga_log)

        print("\n=== PROSES OPTIMASI GA ===")
        print(df_ga_results)


        # =========================================================
        # BEST PARAMETER
        # =========================================================
        solution, fitness, _ = ga.best_solution()

        best_C, best_gamma, best_eps = solution

        print("\n=== BEST PARAMETERS ===")
        print("C       =", best_C)
        print("gamma   =", best_gamma)
        print("epsilon =", best_eps)


        # =========================================================
        # MODEL FINAL SVR-GA
        # =========================================================
        svr = SVR(
            kernel="rbf",
            C=best_C,
            gamma=best_gamma,
            epsilon=best_eps
        ) 
    else :
        svr = SVR(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        epsilon=0.1
        )        

    # else:
    #     raise ValueError("model_type tidak dikenali")
        
    svr.fit(X_train_scaled, y_train_scaled)
    
    # Predictions
    y_train_pred_scaled = svr.predict(X_train_scaled)
    y_test_pred_scaled = svr.predict(X_test_scaled)
    
    # Denormalize
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    results = {
        'model': svr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape' : train_mape,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_mape' : test_mape,
        'time': training_time
    }
    
    return results

# ===============================
# SIDEBAR - SIMPLIFIED
# ===============================
with st.sidebar:
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        try:
            st.session_state.data = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success("✅ File berhasil diupload")
        except:
            st.error("❌ Error membaca file")
            st.session_state.data = create_sample_data()
    else:
        st.session_state.data = create_sample_data()
        st.info("📌 Menggunakan sample data")
    
    st.markdown("---")
    
    st.markdown("### 🌶️ Jenis Cabai")

    # --- Pastikan session state ada default ---
    if 'selected_cabai' not in st.session_state:
        st.session_state.selected_cabai = "Cabai Rawit"  # default pertama kali
    if 'run_model' not in st.session_state:
        st.session_state.run_model = False  
    if 'results' not in st.session_state:
        st.session_state.results = None

    # --- Selectbox dengan key unik ---
    new_selected_cabai = st.selectbox(
    "Pilih Jenis Cabai",
    ["Cabai Rawit", "Cabai Merah Besar", "Cabai Merah Keriting"],
    index=["Cabai Rawit", "Cabai Merah Besar", "Cabai Merah Keriting"].index(st.session_state.selected_cabai),
    key="select_cabai"
    )
    
    # --- Reset otomatis jika user ganti cabai ---
    if new_selected_cabai != st.session_state.selected_cabai:
        st.session_state.run_model = False
        st.session_state.results = None

    # --- Update session state ---
    st.session_state.selected_cabai = new_selected_cabai

    # Validasi kolom sebelum akses
    if st.session_state.selected_cabai in st.session_state.data.columns:
        cabai_valid = True
        st.write(f"Menampilkan data untuk: **{st.session_state.selected_cabai}**")
        # st.dataframe(st.session_state.data[[st.session_state.selected_cabai]])
    else:
        cabai_valid = False
        st.warning(f"⚠️ Data yang diunggah tidak memuat kolom **{st.session_state.selected_cabai}**")
    
    st.markdown("---")

    # --- Model Selection ---
    st.markdown("### 🤖 Model Prediksi")
    st.markdown("Pilih model yang ingin digunakan")

    model_type = st.selectbox(
        "Model",
        ["SVR", "SVR-GA (Genetic Algorithm)"],
        index=0,
        key="select_model"
        )
        
    st.markdown("---")

    # --- Actions ---
    st.markdown("### 🚀 Jalankan Analisis")
   
    # Cek NA di kolom yang dipilih
    if 'data' in st.session_state and st.session_state.selected_cabai in st.session_state.data.columns:
        na_exists = st.session_state.data[st.session_state.selected_cabai].isna().any()
    else:
        na_exists = True  # disable tombol jika kolom tidak ada / data belum ada

    if na_exists:
        st.warning(f"⚠️ Kolom '{st.session_state.selected_cabai}' memiliki nilai NA. Lengkapi data sebelum menjalankan analisis.")

    # Tombol Jalankan Model
    if st.button("🚀 Jalankan Model", use_container_width=True, disabled=na_exists):
        st.session_state.run_model = True
    
    st.markdown("---")
    
    # Download button
    #st.download_button(
    #    label="📥 Unduh Prediksi",
    #    data="Data prediksi akan tersedia setelah model selesai dijalankan",
    #    file_name="prediksi.csv",
    #    mime="text/csv",
    #    use_container_width=True
    #)

# ===============================
# MAIN CONTENT
# ===============================
data = st.session_state.data

cabai_map = {
    "Cabai Rawit": "rawit",
    "Cabai Merah Besar": "merah_besar",
    "Cabai Merah Keriting": "merah_keriting"
}

#col = cabai_map[selected_cabai]
#series = st.session_state.data[col].values

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tampilan Impor Data",
    "Visualisasi Pola Deret Waktu",
    "Penyiapan Data & Evaluasi Model",
    "Visualisasi Hasil Prediksi",
    "Data Hasil Prediksi"
])

# ======================== TAB 1 ========================
with tab1:
    st.markdown("### 💾 Data yang Diimpor")
    
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 2])
    with col1:
        st.metric("Total Baris", len(data))
    with col2:
        st.metric("Total Kolom", len(data.columns))
    with col3:
        st.metric("Date Range", f"{len(data)} hari")
    with col4:
        na_count = data.isna().sum().sum()
        if na_count > 0:
            st.metric("Status", f"❌ Tidak Valid") #  ({na_count} NA)
        else:
            st.metric("Status", "✅ Valid")
        # st.metric("Status", "✅ Valid")
    
    st.markdown("---")
    st.markdown("#### Pratinjau Data")
    st.dataframe(data, use_container_width=True)
    
    desc = data.iloc[:, 1:].describe().round(2)

    desc.rename(index={
        "count": "Banyaknya Data",
        "mean": "Rata-rata",
        "std": "Simpangan Baku",
        "min": "Minimum",
        "25%": "Kuartil 1 (Q1)",
        "50%": "Median (Q2)",
        "75%": "Kuartil 3 (Q3)",
        "max": "Maksimum"
        }, inplace=True)
    
    if not cabai_valid:
        st.warning(
        f"⚠️ Data yang diunggah tidak memuat kolom **{st.session_state.selected_cabai}**"
        )

    st.markdown("---")
    if cabai_valid:
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(desc, use_container_width=True)
    
    # cabai_valid = selected_cabai in data.columns    


# ======================== TAB 2 ========================
with tab2:
    st.markdown("### 📈 Visualisasi Pola Deret Waktu")
    
    
    if st.session_state.selected_cabai in data.columns and cabai_valid == True:
        values = data[st.session_state.selected_cabai]
        
        # Cek apakah ada NA
        if values.isna().any() or values.shape[0] == 0:
            st.warning("⚠️ Data pada Variabel yang Digunakan Tidak Lengkap")
        else:
            try:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rata-rata", f"Rp {values.mean():,.0f}")
                with col2:
                    st.metric("Minimum", f"Rp {values.min():,.0f}")
                with col3:
                    st.metric("Maximum", f"Rp {values.max():,.0f}")
                with col4:
                    st.metric("Std Dev", f"Rp {values.std():,.0f}")
                
                st.markdown("---")
                st.markdown(f"#### Harga {st.session_state.selected_cabai} per Tanggal")
                
                # Plot time series
                fig = plot_time_series_interactive(data, st.session_state.selected_cabai)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
            except ValueError as e:
                st.error(f"⚠️ Terjadi error saat visualisasi: {e}")

    else:
        st.warning(
        f"⚠️ Data yang diunggah tidak memuat kolom **{st.session_state.selected_cabai}**"
        )



# ======================== TAB 3 ========================
with tab3:
    st.markdown("### 📊 Penyiapan Data")
    
    # Prepare data
    # series_data = data[selected_cabai].values
            
    # Create supervised dataset
    if st.session_state.run_model:    
        data = st.session_state.data
        series = data[st.session_state.selected_cabai].values
        n = len(series)
            
        # =========================
        # SPLITTING OPTION
        # =========================
        # st.subheader("Data Splitting")
        st.markdown('#### *Data Splitting*')

        split_mode = st.radio(
        "Silakan Pilih Metode *Data Splitting* (Pembagian Data)",
        ["Persentase", "Input Langsung"]
        )
    
        if split_mode == "Persentase":
            pct = st.number_input("Persentase Data Training (%)",
            min_value=1, max_value=99, value=80)
            test_size = int(n * (100 - pct) / 100)
        else:
            test_size = st.number_input("Jumlah Data Testing",
            min_value=1, max_value=n-1, value=30)
        train_size = n - test_size
        train_raw = series[:train_size]
        
        st.write(f"📌 Train: {train_size} | Test: {test_size}")

        st.markdown("---")
    
        # =========================
        # PACF TRAINING
        # =========================
        # st.subheader("PACF pada Data *Training*")
        st.markdown('#### PACF Data Training')

        max_lag = st.slider("Maksimum lag PACF",
        min_value = 1, max_value=min(40, train_size//2), value=20)

        pacf_vals = pacf(train_raw, nlags=max_lag, method="ywm")
        Z = scipy.stats.norm.ppf(0.975)
        conf = Z / np.sqrt(len(train_raw))
    
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(train_raw, lags=max_lag, ax=ax, alpha=0.05)
    
        ax.set_title(f'PACF - {st.session_state.selected_cabai}', fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('PACF')
        ax.grid(True, alpha=0.3)
    
        st.pyplot(fig)

        pacf_sig = np.where(np.abs(pacf_vals) > conf)[0]
        pacf_sig = pacf_sig[pacf_sig > 0]

        # st.write("Lag signifikan (PACF):", pacf_sig.tolist())
        # st.write("Lag signifikan (PACF):", pacf_sig)
            
        def render_lag_text(lags):
            lags = list(map(int, lags))
            return ", ".join(map(str, lags))
                
    #st.markdown(
    #    f"**Lag signifikan (PACF):** {render_lag_text(pacf_sig)}"
    #    )
            
        st.markdown(
        "**Lag signifikan (PACF):** " +
        " ".join([f"`{lag}`" for lag in pacf_sig])
        )

        st.markdown("---")

        # =========================
        # LAG FEATURE OPTION
        # =========================
    # st.subheader("Pemilihan Lag Feature")
        st.markdown('#### Pemilihan *Lag Feature*')

        lag_mode = st.radio(
        "Metode pemilihan lag",
        ["PACF signifikan", "Input manual"]
        )

        if lag_mode == "PACF signifikan":
            lags = pacf_sig
        else:
            max_manual_lag = st.number_input(
            "Banyak lag",
            min_value=1, max_value=train_size-1, value=1
            )
            lags = np.arange(1, max_manual_lag + 1)

        lag_notation = ", ".join([rf"Y_{{t-{i}}}" for i in lags])
    
        st.markdown("###### 🔎 Fitur Lag yang Digunakan")
        st.markdown(rf"${lag_notation}$")

    # st.write("Lag yang digunakan:", lags.tolist())

    # =========================
    # BUILD LAG FEATURES
    # =========================
        series_all = series  # train + test digabung sementara
        df = pd.DataFrame({"y": series_all})

    # df = df.copy()
        for l in lags:
            df[f"lag_{l}"] = df["y"].shift(l)
        
        df_NaN = df.copy()
        df = df.dropna().reset_index(drop=True)
    
        st.markdown("---")

    # =========================
    # FINAL SPLIT (TEST SIZE FIX)
    # =========================
        X = df.drop(columns="y").values
        y = df["y"].values

        X_train = X[:-test_size]
        X_test  = X[-test_size:]
        y_train = y[:-test_size]
        y_test  = y[-test_size:]

        st.markdown('#### Dimensi dan Pratinjau Data yang Akan Dimodelkan')
    
        st.markdown(f"- *Training Data* (X) : `{X_train.shape}`")
        st.markdown(f"- *Training Data* (Y) : `{y_train.shape}`")
        st.markdown(f"- *Testing Data* (X) : `{X_test.shape}`")
        st.markdown(f"- *Testing Data* (Y) : `{y_test.shape}`")
    
    

        if st.checkbox("Tampilkan Data Setelah Pembuatan *Lag Feature*"):
            st.dataframe(df_NaN)
        if st.checkbox("Tampilkan Data Setelah Penghapusan NaN"):
            st.dataframe(df)
    
    # =========================
    # OUTPUT DIMENSI
    # =========================
    # st.subheader("Dimensi Data Siap Model")
    # st.markdown('#### Dimensi Data Siap Model')

    #df_info = pd.DataFrame({
    #"Data": ["Training Data (X)", "Training Data (Y)", "Testing Data (X)", "Testing Data (Y)"],
    #"Dimensi": [X_train.shape, y_train.shape, X_test.shape, y_test.shape]
    #})
    
    #st.table(df_info)
    
    # ======================== TAB 3 ======================== with tab3:
        st.markdown("---")
    
        st.markdown("### 🎯 Evaluasi Model")
    
        with st.spinner("⏳ Model sedang dijalankan..."):
            # Train model
            model_name = "SVR" if "SVR" in model_type else "SVR-GA (Genetic Algorithm)"
            if model_type == "SVR":
                results = train_svr_model(X_train, X_test, y_train, y_test, model_type = "SVR")
            else:
                results = train_svr_model(X_train, X_test, y_train, y_test, model_type = "SVR-GA (Genetic Algorithm)")
            #results = train_svr_model(X_train, X_test, y_train, y_test, 
            #                         model_type="SVR" if "SVR" in model_type else "SVR-GA (Genetic Algorithm)")
            
            st.session_state.results = results
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.model_type = model_name
        
        st.success("✅ Model selesai dijalankan!")
        
        st.markdown(f"#### Metrik Evaluasi Model ({model_type})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set**")
            train_metrics = pd.DataFrame({
                'Metrik': ['RMSE', 'MAE', 'R²', 'MAPE'],
                'Nilai': [
                    f"Rp {results['train_rmse']:,.2f}",
                    f"Rp {results['train_mae']:,.2f}",
                    f"{results['train_r2']:.4f}",
                    f"{results['train_mape']* 100:.4f}%"
                ]
            })
            st.dataframe(train_metrics, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Test Set**")
            test_metrics = pd.DataFrame({
                'Metrik': ['RMSE', 'MAE', 'R²', 'MAPE'],
                'Nilai': [
                    f"Rp {results['test_rmse']:,.2f}",
                    f"Rp {results['test_mae']:,.2f}",
                    f"{results['test_r2']:.4f}",
                    f"{results['test_mape']* 100:.4f}%"
                ]
            })
            st.dataframe(test_metrics, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown(f"⏱️ **Training Time:** {results['time']:.3f} detik")
        
        st.markdown("---")
        st.markdown("#### Grafik Perbandingan Metrik")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_names = ['RMSE', 'MAE']
            train_vals = [results['train_rmse'], results['train_mae']]
            test_vals = [results['test_rmse'], results['test_mae']]
            
            x = np.arange(len(metrics_names))
            width = 0.45
            
            bars1 = ax.bar(x - width/2, train_vals, width, label='Training', color='#667eea', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='#764ba2', alpha=0.8)
            
            ax.set_xlabel('Metrik', fontweight='bold')
            ax.set_ylabel('Nilai')
            ax.set_title('Metrik RMSE dan MAE pada Training vs Testing Data')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_names = ['R²', 'MAPE']
            train_vals = [results['train_r2']*100, results['train_mape']*100]
            test_vals = [results['test_r2']*100, results['test_mape']*100]
            
            x = np.arange(len(metrics_names))
            width = 0.45
            
            bars1 = ax.bar(x - width/2, train_vals, width, label='Training', color='#667eea', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='#764ba2', alpha=0.8)
            
            ax.set_xlabel('Metrik', fontweight='bold')
            ax.set_ylabel('Nilai (%)')
            ax.set_title('Metrik R² dan MAPE pada Training vs Testing Data')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
        
        #with col2:
        #    fig, ax = plt.subplots(figsize=(8, 5))
        #    r2_vals = [results['train_r2'], results['test_r2']]
        #    sets = ['Training', 'Test']
        #    colors = ['#667eea', '#764ba2']
            
        #   bars = ax.bar(sets, r2_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        #   ax.set_ylabel('R² Score', fontweight='bold')
        #    ax.set_title('R² Score Comparison')
        #   ax.set_ylim([0, 1])
        #   ax.grid(True, alpha=0.3, axis='y')
            
        #    for bar, val in zip(bars, r2_vals):
        #       height = bar.get_height()
        #       ax.text(bar.get_x() + bar.get_width()/2., height,
        #              f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
        #   st.pyplot(fig)
    
    else:
        st.info("ℹ️ Klik Tombol **Jalankan Model** di Sidebar untuk Melatih Model")


# ======================== TAB 4 ========================
with tab4:
    st.markdown("### 📉 Visualisasi Hasil Prediksi")
    
    if st.session_state.run_model and st.session_state.results:
        results = st.session_state.results
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        y_train_pred = results['train_pred']
        y_test_pred = results['test_pred']
        
        all_actual = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=all_actual,
            name='Harga Aktual',
            mode='lines',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='<b>Aktual:</b> Rp %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            y=all_pred,
            name='Prediksi',
            mode='lines',
            line=dict(color='#4ECDC4', width=2, dash='dash'),
            hovertemplate='<b>Prediksi:</b> Rp %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Perbandingan Harga Aktual vs Prediksi",
            xaxis_title="Index",
            yaxis_title="Harga (Rp)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            plot_bgcolor='rgba(240,240,240,0.5)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Analisis Residual")
        
        residuals = all_actual - all_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(residuals, linewidth=2, color='#667eea')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax.fill_between(range(len(residuals)), residuals, alpha=0.3, color='#667eea')
            ax.set_title('Residual Over Time', fontweight='bold')
            ax.set_ylabel('Residual (Rp)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
            ax.set_title('Distribusi Residual', fontweight='bold')
            ax.set_xlabel('Residual (Rp)')
            ax.set_ylabel('Frekuensi')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
    
    else:
        st.info("ℹ️ Klik Tombol **Jalankan Model** di Sidebar Terlebih Dahulu")


# ======================== TAB 5 ========================
with tab5:
    st.markdown("### 📋 Data Hasil Prediksi")
    
    if st.session_state.run_model and st.session_state.results:
        results = st.session_state.results
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        y_train_pred = results['train_pred']
        y_test_pred = results['test_pred']
        
        train_df = pd.DataFrame({
            'Index': range(len(y_train)),
            'Set': 'Training',
            'Aktual': y_train.astype(int),
            'Prediksi': y_train_pred.astype(int),
            'Error': (y_train - y_train_pred).astype(int),
            'Error %': ((y_train - y_train_pred) / y_train * 100).round(2)
        })
        
        test_df = pd.DataFrame({
            'Index': range(len(y_test)),
            'Set': 'Test',
            'Aktual': y_test.astype(int),
            'Prediksi': y_test_pred.astype(int),
            'Error': (y_test - y_test_pred).astype(int),
            'Error %': ((y_test - y_test_pred) / y_test * 100).round(2)
        })
        
        results_df = pd.concat([train_df, test_df], ignore_index=True)
        
        st.dataframe(results_df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name=f"prediksi_{st.session_state.selected_cabai}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### Ringkasan Hasil")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Set**")
            st.write(f"Total Sampel: {len(train_df)}")
            st.write(f"*Mean Absolute Error*: Rp {train_df['Error'].abs().mean():,.2f}")
            st.write(f"*Mean Absolute Percentage Error* (%): {train_df['Error %'].abs().mean():.2f}%")
        
        with col2:
            st.markdown("**Test Set**")
            st.write(f"Total Sampel: {len(test_df)}")
            st.write(f"*Mean Absolute Error*: Rp {test_df['Error'].abs().mean():,.2f}")
            st.write(f"*Mean Absolute Percentage Error* (%): {test_df['Error %'].abs().mean():.2f}%")
    
    else:
        st.info("ℹ️ Klik Tombol **Jalankan Model** di Sidebar Terlebih Dahulu")