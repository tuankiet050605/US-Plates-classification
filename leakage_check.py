# 1. Import libraries
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

DATA_DIR = "data/brain_mri(clean_dataset)"
OUT_ROOT = "data/model_outputs"
OUT_DIR = os.path.join(OUT_ROOT, "leakage")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_GROUPS = {
    "simple": os.path.join(OUT_ROOT, "simple_cnn", "run_*", "*.keras"),
    "complex": os.path.join(OUT_ROOT, "complex_cnn", "run_*", "*.keras"),
    "transfer": os.path.join(OUT_ROOT, "transfer_resnet50", "run_*", "*.keras"),
}
HIGH_T = 0.85
BANDS = [0, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
CLASSES = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# 2. Load ảnh
def load(split, size):
    X, y = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, name)
        for f in sorted(os.listdir(folder)):
            X.append(np.asarray(Image.open(os.path.join(folder, f)).convert("L").resize((size, size)), dtype="float32"))
            y.append(label)
    return np.stack(X), np.array(y)


def vec(X):
    V = X.reshape(len(X), -1)
    V = V - V.mean(1, keepdims=True)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)


xtr, ytr = load("train", 128)
xte, yte = load("test", 128)
x224, _ = load("test", 224)
x224 = x224[..., None] / 255.0
print("Train:", len(xtr), "| Test:", len(xte))

# 3. Corr test - train (có lật), 1-NN pixel
A, B, BF = vec(xte), vec(xtr), vec(xtr[:, :, ::-1])
S = np.maximum(A @ B.T, A @ BF.T)
max_corr = S.max(1)
nn_pred = ytr[S.argmax(1)]
del A, B, BF, S

df = pd.DataFrame({"label": [CLASSES[i] for i in yte], "max_corr": max_corr, "1nn_ok": nn_pred == yte})
df["band"] = pd.cut(df.max_corr, BANDS)
high = (df.max_corr > HIGH_T).to_numpy()
print(f"Ảnh test có corr > {HIGH_T}: {high.sum()} | còn lại: {(~high).sum()} | max corr: {max_corr.max():.4f}")

# 4. Dự đoán của từng mô hình, từng lần chạy
model_cols = {}
for group, pattern in MODEL_GROUPS.items():
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("Không tìm thấy mô hình:", pattern)
        continue
    for path in paths:
        run = os.path.basename(os.path.dirname(path))
        col = f"{group}_{run}"
        K.clear_session()
        model = load_model(path)
        pred = model.predict(x224, batch_size=32, verbose=0).argmax(1)
        df[col] = pred == yte
        model_cols[col] = group
        print(f"{col}: acc = {df[col].mean():.4f}")
K.clear_session()


# 5. Tổng hợp theo từng lần chạy
def row(name, group, ok):
    acc, acc_high, acc_low = ok.mean(), ok[high].mean(), ok[~high].mean()
    return {"model": name, "group": group, "acc": acc, "acc_corr_high": acc_high, "acc_corr_low": acc_low,
            "chenh": acc_high - acc_low, "tang_toi_da": acc - acc_low}


rows = [row("1-NN pixel", "1-NN pixel", df["1nn_ok"].to_numpy())]
rows += [row(col, group, df[col].to_numpy()) for col, group in model_cols.items()]
per_run = pd.DataFrame(rows).set_index("model")
print(f"\n========== Từng lần chạy (corr > {HIGH_T}: n = {high.sum()}) ==========")
print(per_run.drop(columns="group").round(4))
per_run.round(4).to_csv(os.path.join(OUT_DIR, "leakage_per_run.csv"))

# 6. Tổng hợp theo mô hình (trung bình ± độ lệch các lần chạy)
metrics = ["acc", "acc_corr_high", "acc_corr_low", "chenh", "tang_toi_da"]
per_model = per_run.groupby("group", sort=False)[metrics].agg(["mean", "std"]).round(4)
per_model.insert(0, ("n_runs", ""), per_run.groupby("group", sort=False).size())
print("\n========== Theo mô hình (mean ± std) ==========")
print(per_model)
per_model.to_csv(os.path.join(OUT_DIR, "leakage_per_model.csv"))

# 7. Độ chính xác theo mốc corr (trung bình các lần chạy của mỗi mô hình)
band_table = df.groupby("band", observed=True).agg(n=("label", "size"), **{"1-NN pixel": ("1nn_ok", "mean")})
for group in dict.fromkeys(model_cols.values()):
    cols = [c for c, g in model_cols.items() if g == group]
    band_table[group] = df.groupby("band", observed=True)[cols].mean().mean(axis=1)
print("\n========== Accuracy theo mốc corr ==========")
print(band_table.round(3))
band_table.round(4).to_csv(os.path.join(OUT_DIR, "leakage_bands.csv"))

# 8. Số ảnh test theo lớp x mốc corr
counts = pd.crosstab(df.label, df.band)
print("\n========== Số ảnh test theo lớp x mốc corr ==========")
print(counts)
counts.to_csv(os.path.join(OUT_DIR, "leakage_counts.csv"))

print("\nĐã lưu kết quả vào", OUT_DIR)
