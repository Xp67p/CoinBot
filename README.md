Coin Bot – CNN-Based Jordanian Coin Classification

This project implements a convolutional neural network (CNN) to classify Jordanian coin denominations (5 piasters, 10 piasters, 25 piasters, and 1/2 dinar) using supervised deep learning. The model is trained using transfer learning with MobileNetV2 and evaluated on a structured dataset of coin images.

The trained model is deployed in a simple real-time pipeline where an image is received, classified by the CNN, and the predicted label is sent back to an embedded device for physical coin sorting. The focus of this repository is on model training, fine-tuning, and evaluation, rather than hardware design.

The final model achieves ~82% classification accuracy, with most misclassifications occurring between visually similar low-denomination coins (5 and 10 piasters).
