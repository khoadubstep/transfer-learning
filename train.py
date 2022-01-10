from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import halo.config as cfg
import numpy as np
import pickle
import os


def load_data_split(splitPath):
    data = []
    labels = []

    for row in open(splitPath):
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype="float")

        data.append(features)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)


trainPath = os.path.sep.join([cfg.OUTPUT_PATH, "{}.csv".format(cfg.TRAIN)])
testPath = os.path.sep.join([cfg.OUTPUT_PATH, "{}.csv".format(cfg.TEST)])

print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainPath)
(testX, testY) = load_data_split(testPath)

le = pickle.loads(open(cfg.ENCODER_PATH, "rb").read())

print("[INFO] training model...")
model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=150)
model.fit(trainX, trainY)

print("[INFO] evaluating...")
predY = model.predict(testX)
print(classification_report(testY, predY, target_names=le.classes_))

print("[INFO] saving model...")
f = open(cfg.MODEL_PATH, "wb")
f.write(pickle.dumps(model))
f.close()
