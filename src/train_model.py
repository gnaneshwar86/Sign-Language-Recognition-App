import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

FRAME_DIR = "dataset/frames"

X, y = [], []
labels = sorted(os.listdir(FRAME_DIR))
label_map = {label: i for i, label in enumerate(labels)}

# Load images
for label in labels:
    for img in os.listdir(os.path.join(FRAME_DIR, label)):
        path = os.path.join(FRAME_DIR, label, img)
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        X.append(image)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
output = Dense(len(labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/asl_model.h5")

print("✅ Model training completed and saved")
