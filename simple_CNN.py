# 1. Import libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data/brain_mri(clean_dataset)"
OUT_DIR = "data/model_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
NUM_CLASSES = len(CLASSES)


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

# 3. Normalize (scale pixel về [0,1])
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 4. Data Augmentation (chỉ áp dụng cho train)
def random_brightness(x):
    return np.clip(x * np.random.uniform(0.85, 1.15), 0.0, 1.0)

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="constant",
    cval=0.0,
    preprocessing_function=random_brightness,
)
train_gen = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
class_weight = dict(enumerate(weights))
print("Class weight:", {CLASSES[k]: round(v, 3) for k, v in class_weight.items()})

# 5. Build simple CNN model
model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))
model.summary()

# 6. Compile
opt = Adam(learning_rate=1e-3)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 7. Train
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(OUT_DIR, "simple_cnn_mri.keras"), monitor="val_loss", save_best_only=True)
]
H = model.fit(
    train_gen,
    validation_data=(x_val, y_val),
    epochs=60,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# 8. Plot learning curves
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="valid")
plt.title("accuracy"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(H.history["loss"], label="train")
plt.plot(H.history["val_loss"], label="valid")
plt.title("loss"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "simple_cnn_mri_learning_curves.png"))
plt.close()
print("Đã lưu hình đường học")

# 9. Evaluate
print("Đang đánh giá trên tập test...")
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred_all = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
print("Test F1 macro:", f1_score(y_test, y_pred_all, average="macro"))
print(classification_report(y_test, y_pred_all, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(y_test, y_pred_all)
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks(range(NUM_CLASSES), CLASSES, rotation=30)
plt.yticks(range(NUM_CLASSES), CLASSES)
plt.xlabel("dự đoán"); plt.ylabel("thật"); plt.title("Confusion matrix (test)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "simple_cnn_mri_confusion_matrix.png"))
plt.close()

# So sánh train / valid / test (ảnh gốc, không augment, Dropout tắt)
rows = []
for split, x, y in [("train", x_train, y_train), ("valid", x_val, y_val), ("test", x_test, y_test)]:
    loss, acc = model.evaluate(x, y, batch_size=BATCH_SIZE, verbose=0)
    pred = np.argmax(model.predict(x, batch_size=BATCH_SIZE, verbose=0), axis=1)
    rows.append({"split": split, "loss": loss, "accuracy": acc, "f1_macro": f1_score(y, pred, average="macro")})
compare = pd.DataFrame(rows).set_index("split").round(4)
compare.loc["train - test"] = (compare.loc["train"] - compare.loc["test"]).round(4)
print(compare)
compare.to_csv(os.path.join(OUT_DIR, "simple_cnn_mri_train_vs_test.csv"))

# 10. Predict one sample image
plt.imshow(x_test[0, ..., 0], cmap="gray")
plt.axis("off")
plt.savefig(os.path.join(OUT_DIR, "simple_cnn_mri_sample.png"))
plt.close()

y_one = model.predict(x_test[:1], verbose=0)
print("Giá trị dự đoán:", CLASSES[np.argmax(y_one)], "· Giá trị thật:", CLASSES[y_test[0]])

# 11. Save results
results = {
    "test_loss": float(score[0]),
    "test_accuracy": float(score[1]),
    "test_f1_macro": float(f1_score(y_test, y_pred_all, average="macro")),
    "epochs_run": len(H.history["loss"]),
    "best_epoch": int(np.argmin(H.history["val_loss"])) + 1,
}
with open(os.path.join(OUT_DIR, "simple_cnn_mri_results.json"), "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame(H.history).to_csv(os.path.join(OUT_DIR, "simple_cnn_mri_history.csv"), index_label="epoch")

report = classification_report(y_test, y_pred_all, target_names=CLASSES, zero_division=0, output_dict=True)
pd.DataFrame(report).T.to_csv(os.path.join(OUT_DIR, "simple_cnn_mri_per_class.csv"))

print("Đã lưu kết quả vào", OUT_DIR)