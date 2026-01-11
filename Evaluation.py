import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

# paths
DATA_PATH = r"C:\Users\Xp677\Desktop\University\Machine Learning\CoinBot\Data\proccessed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# load model
model = load_model("coin_model.keras")
# load validation data
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
val_data = datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
# predictions
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())
# accuracy
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", round(accuracy * 100, 2))

# classification report
print(classification_report(y_true, y_pred, target_names=labels))

# colored confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
