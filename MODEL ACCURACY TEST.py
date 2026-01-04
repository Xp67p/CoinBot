import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


DATA_PATH = r"C:\Users\Xp677\Desktop\University\Machine Learning\ML_DL PROJECT\Data\proccessed"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# load dataset (same split style as training)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

# load trained model
model = tf.keras.models.load_model("coin_model.keras")

# predict
preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

# accuracy
acc = np.mean(y_pred == y_true)
print("validation accuracy:", round(acc * 100, 2), "%")

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nconfusion matrix:")
print(cm)

# class names
labels = list(val_data.class_indices.keys())

print("\nclassification report:")
print(classification_report(y_true, y_pred, target_names=labels))
