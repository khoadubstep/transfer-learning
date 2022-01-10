from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import halo.config as cfg
from imutils import paths
import numpy as np
import pickle
import random
import os

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None

if not os.path.exists(cfg.OUTPUT_PATH):
    os.makedirs(cfg.OUTPUT_PATH)

for split in (cfg.TRAIN, cfg.TEST, cfg.VAL):
    print("[INFO] processing '{} split'...".format(split))
    splitPath = os.path.sep.join([cfg.GEN_PATH, split])
    imagePaths = list(paths.list_images(splitPath))

    random.shuffle(imagePaths)
    labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]

    if le is None:
        le = LabelEncoder()
        le.fit(labels)

    csvPath = os.path.sep.join([cfg.OUTPUT_PATH, "{}.csv".format(split)])
    csv = open(csvPath, "w")

    for (b, i) in enumerate(range(0, len(imagePaths), cfg.BATCH_SIZE)):
        print("[INFO] processing batch {}/{}".format(b + 1,
              int(np.ceil(len(imagePaths) / float(cfg.BATCH_SIZE)))))
        batchPaths = imagePaths[i:i + cfg.BATCH_SIZE]
        batchLabels = le.transform(labels[i:i + cfg.BATCH_SIZE])
        batchImages = []

        for imagePath in batchPaths:
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)

            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=cfg.BATCH_SIZE)
        features = features.reshape((features.shape[0], 7 * 7 * 512))

        for (label, vec) in zip(batchLabels, features):
            vec = ",".join([str(v) for v in vec])
            csv.write("{},{}\n".format(label, vec))

    csv.close()

f = open(cfg.ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
