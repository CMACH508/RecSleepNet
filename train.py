import numpy as np
import tensorflow as tf
from data import get_split
from model import TinySleepNet
from minibatching import batch_generator


import logging
tf.get_logger().setLevel(logging.ERROR)


def train(
    config,
    fold_idx,
    output_dir,
    log_file,
    restart=False,
    random_seed=42,
):
    # Create output directory for the specified fold_idx
    train_x,train_y,valid_x,valid_y,test_x,test_y=get_split(config,fold_idx,random_seed)

    # Create a model
    model = TinySleepNet(
        config=config,
        use_best=False,
        use_rec=True
    )
    best_acc = 0
    best_f1 = 0
    if model.use_rec:
        for epoch in range(config["pretrain_epochs"]):
            print("pretrain_epoch:{}".format(epoch))
            shuffle_idx = np.random.permutation(np.arange(len(train_x)))
            train_minibatch_fn = batch_generator(train_x,train_y,config,shuffle_idx=shuffle_idx)
            model.train(train_minibatch_fn)
            valid_minibatch_fn = batch_generator(valid_x,valid_y,config)
            model.train(valid_minibatch_fn)
            test_minibatch_fn = batch_generator(test_x,test_y,config)
            model.train(test_minibatch_fn)
    model.pretrain = False
    for epoch in range(config["n_epochs"]):
        print("epoch:{}".format(epoch))
        shuffle_idx = np.random.permutation(np.arange(len(train_x)))
        train_minibatch_fn = batch_generator(train_x,train_y,config,shuffle_idx=shuffle_idx)
        train_acc,train_f1,loss = model.train(train_minibatch_fn)
        print("Train: Acc:{:.4f}, F1:{:.4f}, loss:{:.4f}".format(train_acc,train_f1,loss))
        valid_minibatch_fn = batch_generator(valid_x, valid_y, config)
        valid_acc,valid_f1,_ = model.evaluate(valid_minibatch_fn)
        print("Valid: Acc:{:.4f}, F1:{:.4f}".format(valid_acc,valid_f1))
        test_minibatch_fn = batch_generator(test_x, test_y, config)
        test_acc,test_f1,_ = model.evaluate(test_minibatch_fn)
        print("Valid: Acc:{:.4f}, F1:{:.4f}".format(test_acc, test_f1))
        if valid_acc > best_acc and valid_f1 > best_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            model.save_best_checkpoint(name="best_model")
