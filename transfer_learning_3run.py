# 1. Import libraries
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Concatenate, Rescaling, Normalization, BatchNormalization,
                                     GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data/brain_mri(clean_dataset)"
BACKBONE = "resnet50"
OUT_DIR = f"data/model_outputs/transfer_{BACKBONE}"
os.makedirs(OUT_DIR, exist_ok=True)
RUNS = [1, 2, 3]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FT = 30
LR_HEAD = 1e-3
LR_FT = 5e-5
LABEL_SMOOTHING = 0.1
CLASSES = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
NUM_CLASSES = len(CLASSES)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# 2. Load dataset (ảnh xám 1 kênh)
def load_split(split):
    X, y = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, name)
        for fname in sorted(os.listdir(folder)):
            img = Image.open(os.path.join(folder, fname)).convert("L").resize((IMG_SIZE, IMG_SIZE))
            X.append(np.asarray(img))
            y.append(label)
    return np.stack(X)[..., None], np.array(y)

x_train, y_train = load_split("train")
x_val, y_val = load_split("valid")
x_test, y_test = load_split("test")

print("Train:", x_train.shape, np.bincount(y_train))
print("Validation:", x_val.shape, np.bincount(y_val))
print("Test:", x_test.shape, np.bincount(y_test))

y_train_oh = to_categorical(y_train, NUM_CLASSES)
y_val_oh = to_categorical(y_val, NUM_CLASSES)
y_test_oh = to_categorical(y_test, NUM_CLASSES)

# 3. Normalize (scale pixel về [0,1])
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 4. Data Augmentation (chỉ áp dụng cho train)
def brightness_contrast(x):
    x = x * np.random.uniform(0.85, 1.15)
    m = x.mean()
    x = (x - m) * np.random.uniform(0.85, 1.15) + m
    return np.clip(x, 0.0, 1.0)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="constant",
    cval=0.0,
    preprocessing_function=brightness_contrast,
)

weights = np.sqrt(compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train))
class_weight = dict(enumerate(weights))
print("Class weight:", {CLASSES[k]: round(v, 3) for k, v in class_weight.items()})

# 5. Build transfer learning model (backbone ImageNet + head mới)
BACKBONES = {
    "resnet50": {"net": ResNet50, "fine_tune_from": "conv5_block1_1_conv"},
    "mobilenetv2": {"net": MobileNetV2, "fine_tune_from": "block_13_expand"},
}


def to_backbone_input(x):
    x = Concatenate(name="gray_to_rgb")([x, x, x])
    if BACKBONE == "resnet50":
        x = Rescaling(255.0, name="to_0_255")(x)
        x = Normalization(mean=[103.939, 116.779, 123.68], variance=[1.0, 1.0, 1.0], name="imagenet_mean")(x)
    else:
        x = Rescaling(2.0, offset=-1.0, name="to_minus1_1")(x)
    return x


def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    base = BACKBONES[BACKBONE]["net"](include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = to_backbone_input(inputs)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs, outputs, name=f"transfer_{BACKBONE}"), base


def make_callbacks(ckpt, patience, best=None, weights_only=False):
    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=weights_only,
                        initial_value_threshold=best),
    ]


def test_scores(model):
    pred = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    return float(np.mean(pred == y_test)), float(f1_score(y_test, pred, average="macro"))


build_model()[0].summary()


# 6-11. Train, evaluate, save cho từng lần chạy
def run_once(run):
    run_dir = os.path.join(OUT_DIR, f"run_{run}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, f"transfer_{BACKBONE}_mri.keras")
    ckpt_head = os.path.join(run_dir, "head_best.weights.h5")
    print(f"\n========== Lần chạy {run} ==========")

    K.clear_session()
    model, base = build_model()

    # 6. Giai đoạn 1: đóng băng backbone, chỉ train head
    model.compile(loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING), optimizer=Adam(LR_HEAD),
                  metrics=["accuracy"])
    print("Giai đoạn 1 - trainable params:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    H1 = model.fit(
        train_datagen.flow(x_train, y_train_oh, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val_oh),
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=make_callbacks(ckpt_head, patience=5, weights_only=True),
        verbose=1
    )
    model.load_weights(ckpt_head)
    head_acc, head_f1 = test_scores(model)
    print(f"Chỉ train head - test accuracy: {head_acc:.4f}, test F1: {head_f1:.4f}")

    # 7. Giai đoạn 2: fine-tune các lớp cuối của backbone (BatchNorm vẫn đóng băng)
    base.trainable = True
    start = [layer.name for layer in base.layers].index(BACKBONES[BACKBONE]["fine_tune_from"])
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, BatchNormalization)

    model.compile(loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING), optimizer=Adam(LR_FT),
                  metrics=["accuracy"])
    print("Giai đoạn 2 - trainable params:", sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    model.save(ckpt)
    best_head = min(H1.history["val_loss"])
    H2 = model.fit(
        train_datagen.flow(x_train, y_train_oh, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val_oh),
        epochs=EPOCHS_FT,
        class_weight=class_weight,
        callbacks=make_callbacks(ckpt, patience=8, best=best_head),
        verbose=1
    )

    K.clear_session()
    model = load_model(ckpt)
    os.remove(ckpt_head)
    history = pd.concat([pd.DataFrame(H1.history), pd.DataFrame(H2.history)], ignore_index=True)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_phase = "fine_tune" if best_epoch > len(H1.history["loss"]) else "head"

    # 8. Plot learning curves
    plt.figure(figsize=(11, 4))
    for k, metric in enumerate(["accuracy", "loss"], 1):
        plt.subplot(1, 2, k)
        plt.plot(history[metric], label="train")
        plt.plot(history[f"val_{metric}"], label="valid")
        plt.axvline(len(H1.history["loss"]) - 0.5, color="gray", linestyle="--", label="bắt đầu fine-tune")
        plt.title(f"{metric} (lần {run})"); plt.xlabel("epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "learning_curves.png"))
    plt.close()

    # 9. Evaluate
    score = model.evaluate(x_test, y_test_oh, batch_size=BATCH_SIZE, verbose=0)
    y_pred_all = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    test_f1 = f1_score(y_test, y_pred_all, average="macro")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("Test F1 macro:", test_f1)
    print(classification_report(y_test, y_pred_all, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_all)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks(range(NUM_CLASSES), CLASSES, rotation=30)
    plt.yticks(range(NUM_CLASSES), CLASSES)
    plt.xlabel("dự đoán"); plt.ylabel("thật"); plt.title(f"Confusion matrix (test, lần {run})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()

    # So sánh train / valid / test (ảnh gốc, không augment, Dropout tắt)
    rows = []
    for split, x, y, y_oh in [("train", x_train, y_train, y_train_oh), ("valid", x_val, y_val, y_val_oh),
                              ("test", x_test, y_test, y_test_oh)]:
        loss, acc = model.evaluate(x, y_oh, batch_size=BATCH_SIZE, verbose=0)
        pred = np.argmax(model.predict(x, batch_size=BATCH_SIZE, verbose=0), axis=1)
        rows.append({"split": split, "loss": loss, "accuracy": acc, "f1_macro": f1_score(y, pred, average="macro")})
    compare = pd.DataFrame(rows).set_index("split").round(4)
    compare.loc["train - test"] = (compare.loc["train"] - compare.loc["test"]).round(4)
    print(compare)
    compare.to_csv(os.path.join(run_dir, "train_vs_test.csv"))

    # 10. Save results
    results = {
        "run": run,
        "backbone": BACKBONE,
        "head_test_accuracy": head_acc,
        "head_test_f1_macro": head_f1,
        "test_loss": float(score[0]),
        "test_accuracy": float(score[1]),
        "test_f1_macro": float(test_f1),
        "train_accuracy": float(compare.loc["train", "accuracy"]),
        "valid_accuracy": float(compare.loc["valid", "accuracy"]),
        "gap_accuracy": float(compare.loc["train - test", "accuracy"]),
        "gap_f1_macro": float(compare.loc["train - test", "f1_macro"]),
        "epochs_head": len(H1.history["loss"]),
        "epochs_fine_tune": len(H2.history["loss"]),
        "best_epoch": best_epoch,
        "best_phase": best_phase,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    history.to_csv(os.path.join(run_dir, "history.csv"), index_label="epoch")

    report = classification_report(y_test, y_pred_all, target_names=CLASSES, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(os.path.join(run_dir, "per_class.csv"))
    print("Đã lưu kết quả lần", run, "vào", run_dir)


for run in RUNS:
    run_once(run)

# 12. Tổng hợp các lần chạy (trung bình ± độ lệch chuẩn)
run_files = sorted(glob.glob(os.path.join(OUT_DIR, "run_*", "results.json")))
summary = pd.DataFrame([json.load(open(p)) for p in run_files]).set_index("run").sort_index()
print("\nGiai đoạn tốt nhất của từng lần:", summary["best_phase"].to_dict())
cols = ["head_test_accuracy", "head_test_f1_macro", "test_accuracy", "test_f1_macro", "test_loss",
        "train_accuracy", "valid_accuracy", "gap_accuracy", "gap_f1_macro",
        "epochs_head", "epochs_fine_tune", "best_epoch"]
summary = summary[cols]
runs_only = summary.copy()
summary.loc["mean"] = runs_only.mean()
summary.loc["std"] = runs_only.std(ddof=1)
print("\n========== Tổng hợp", len(run_files), "lần chạy ==========")
print(summary.round(4))
summary.round(4).to_csv(os.path.join(OUT_DIR, "runs_summary.csv"))

per_class = []
for p in sorted(glob.glob(os.path.join(OUT_DIR, "run_*", "per_class.csv"))):
    df = pd.read_csv(p, index_col=0).loc[CLASSES, ["precision", "recall", "f1-score"]]
    df["run"] = os.path.basename(os.path.dirname(p))
    per_class.append(df)
per_class = pd.concat(per_class).rename_axis("class").reset_index()
per_class_summary = per_class.groupby("class")[["precision", "recall", "f1-score"]].agg(["mean", "std"]).round(3)
print(per_class_summary)
per_class_summary.to_csv(os.path.join(OUT_DIR, "per_class_summary.csv"))

print("Đã lưu tổng hợp vào", OUT_DIR)