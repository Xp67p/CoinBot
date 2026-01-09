import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from pathlib import Path

class CoinModel:
    def __init__(self):
        root = Path(__file__).parent
        self.model = load_model(root / "coin_model.keras")

        self.labels = [
            "50",
            "25",
            "10",
            "5"
        ]

    def predict_image(self, image):
        img = cv2.resize(image, (224, 224))
        img = img.astype("float32")
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        preds = self.model.predict(img)[0]
        return list(zip(preds, self.labels))


def plot_predictions(predictions):
    labels = [p[1] for p in predictions]
    values = [round(p[0], 3) for p in predictions]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")

    return fig, ax
