import argparse
import os
import numpy as np
import tensorflow as tf
from data import get_split
from model import TinySleepNet
from minibatching import batch_generator
from config import config
def compute_performance(cm):
    """Computer performance metrics from confusion matrix.

    It computers performance metrics from confusion matrix.
    It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
    """

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(
    dataset,
    model_dir
):
    cm = np.zeros((5,5))
    config["dataset"] = dataset
    for fold_idx in range(config[dataset]):
        model = TinySleepNet(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            testing=True,
            use_best=True,
        )
        train_x, train_y, valid_x, valid_y, test_x, test_y = get_split(config, fold_idx)
        test_minibatch_fn = batch_generator(test_x, test_y, config)
        test_acc, test_f1,cm_one_fold = model.evaluate(test_minibatch_fn)
        # Get corresponding files
        cm += cm_one_fold
        print("fold={}, acc={:.1f}, mf1={:.1f}".format(
            fold_idx,
            test_acc*100.0,
            test_f1*100.0,
        ))
        tf.reset_default_graph()
    metrics = compute_performance(cm=cm)

    print("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    print("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="isruc")
    parser.add_argument("--dir", type=str, default="./pretrained_model")
    args = parser.parse_args()
    model_dir = os.path.join(args.dir,args.dataset)
    predict(
        dataset = args.dataset,
        model_dir=model_dir,
    )
