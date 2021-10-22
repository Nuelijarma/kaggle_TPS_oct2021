import numpy as np
import pandas as pd

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
    x = data.drop(["target"], axis=1)
    y = data[["target"]]

    logger.debug("Processing training data.")
    logger.debug("(todo)")

    logger.debug("Train/test split.")
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    logger.debug("Training Naive Bayes classifier.")
    clf = GaussianNB()
    clf.fit(train_x, train_y.values.ravel())

    logger.debug("Testing classifier.")
    print(f"test score: {roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])}")

    logger.debug("Cleaning")
    del train_x
    del test_x
    del train_y
    del test_y

    logger.debug("Loading validation data.")
    val_x = pd.read_csv("data/test.csv", index_col="id")
    val_index = val_x.index

    logger.debug("Processing validation data.")
    logger.debug("todo")

    logger.debug("Predicting test 'target' value.")
    val_y = clf.predict_proba(val_x)

    logger.debug("Writing output")
    val_final = pd.DataFrame(data={"id": val_index.values, "target": val_y[:,1]})
    val_final.to_csv("nb_solution.csv", index=False)

if __name__=="__main__":
    main()
