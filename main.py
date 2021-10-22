import numpy as np
import pandas as pd
from tensorflow import keras

# Classifiers
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Misc.
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Debugging
from pdb import set_trace as bp
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    logger.debug("Loading training data.")
    data = pd.read_csv("data/train.csv", index_col="id")
    train_x = data.drop(["target"], axis=1)
    train_y = data[["target"]].astype("category")

    logger.debug("Some statistics")
    n, d = train_x.shape
    logger.debug(f"- {n} items.")
    logger.debug(f"- {d} features")
    nans = train_x.isna().sum().sum()
    logger.debug(f"- {nans} missing values")

    logger.debug("Processing training data")
    logger.debug("- One-hot encoding labels")
    train_y = pd.get_dummies(train_y)

    # logger.debug("Train/test split.")
    # train_x, test_x, train_y, test_y = train_test_split(x, y)

    logger.debug("Training MLP classifier.")
    clf = keras.Sequential([
        keras.layers.Dense(d, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),

    ])
    clf.compile(
        optimizer=keras.optimizers.SGD(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()]
    )
    clf.fit(
        x=train_x.values,
        y=train_y.values,
        batch_size=256,
        validation_split=.2,
        epochs=4,
        verbose=1)


    logger.debug("Cleaning")
    del train_x
    del train_y

    logger.debug("Loading validation data.")
    val_x = pd.read_csv("data/test.csv", index_col="id")
    val_index = val_x.index

    logger.debug("Processing validation data.")
    logger.debug("none")

    logger.debug("Predicting test 'target' value.")
    val_y = clf.predict(
        x=val_x,
        batch_size=256,
        verbose=1)

    logger.debug("Writing output")
    val_final = pd.DataFrame(data={"id": val_index.values, "target": val_y[:,1]})
    val_final.to_csv("mlp_solution.csv", index=False)

if __name__=="__main__":
    main()
