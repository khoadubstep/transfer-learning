import os

ORIG_DATASET = "Food-5K"

GEN_PATH = "generated"

TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

CLASSES = ["non_food", "food"]

BATCH_SIZE = 32

OUTPUT_PATH = "output"
ENCODER_PATH = os.path.sep.join([OUTPUT_PATH, "le.cpickle"])
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "model.cpickle"])
