import socket
import struct
import cv2
import numpy as np
from tensorflow.keras.models import load_model
#Abdullah Aburous
# load trained model
model = load_model("coin_model.keras")

labels = {
    0: "50",
    1: "25",
    2: "10",
    3: "5"
}

reverse_labels = {
    "50": 0,
    "25": 1,
    "10": 2,
    "5": 3
}

HOST = "0.0.0.0"
PORT = 6000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("waiting for image...")

conn, addr = server.accept()
print("connected")

while True:
    # receive image size
    raw_size = conn.recv(4)
    if not raw_size:
        break

    size = struct.unpack("!I", raw_size)[0]

    data = b""
    while len(data) < size:
        packet = conn.recv(1024)
        if not packet:
            break
        data += packet

    # decode image
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # predict
    pred = model.predict(img)[0]
    class_id = int(np.argmax(pred))

    # send result back
    conn.sendall(struct.pack("!i", class_id))

conn.close()
server.close()
