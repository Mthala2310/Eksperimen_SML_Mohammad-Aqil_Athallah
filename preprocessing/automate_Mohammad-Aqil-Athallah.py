import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """Fungsi untuk memuat dataset mentah dari folder raw."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File data mentah tidak ditemukan di: {file_path}")
    print(f"[1/4] Memuat data dari: {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Fungsi otomatisasi preprocessing data sesuai standar eksperimen."""
    print("[2/4] Memulai proses preprocessing data...")
    df_clean = df.copy()

    # 1. Menghapus data duplikat jika ada
    df_clean = df_clean.drop_duplicates()

    # 2. Menghapus kolom identitas yang tidak bernilai prediktif
    columns_to_drop = ["UDI", "Product ID", "Failure Type"]
    df_clean = df_clean.drop(
        columns=[col for col in columns_to_drop if col in df_clean.columns]
    )

    # 3. Encoding Data Kategorikal (Kolom 'Type')
    if "Type" in df_clean.columns:
        le = LabelEncoder()
        df_clean["Type"] = le.fit_transform(df_clean["Type"])
        print("     - Encoding kolom 'Type' selesai.")

    # 4. Normalisasi/Standarisasi Fitur Numerik
    fitur_numerik = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    # Memastikan fitur-fitur tersebut ada di dalam data sebelum di-scale
    fitur_numerik = [col for col in fitur_numerik if col in df_clean.columns]

    if fitur_numerik:
        scaler = StandardScaler()
        df_clean[fitur_numerik] = scaler.fit_transform(df_clean[fitur_numerik])
        print("     - Standarisasi fitur numerik selesai.")

    return df_clean


def save_data(df, output_dir, file_name):
    """Fungsi untuk menyimpan data bersih ke folder tujuan."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)
    print(f"[3/4] Data hasil preprocessing berhasil disimpan di: {output_path}")


if __name__ == "__main__":
    # Menentukan path input (data mentah) dan output (data bersih)
    # Gunakan path relatif yang aman untuk local maupun environment GitHub Actions
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "../namadataset_raw/predictive_maintenance.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
    OUTPUT_FILE = "predictive_maintenance_clean.csv"

    print("=== Memulai Pipeline Otomasi Preprocessing ===")
    try:
        # Jalankan urutan fungsi
        raw_data = load_data(INPUT_PATH)
        cleaned_data = preprocess_data(raw_data)
        save_data(cleaned_data, OUTPUT_DIR, OUTPUT_FILE)
        print("[4/4] Pipeline berjalan sukses 100%!")
    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan pada pipeline: {e}")
