import os
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim
import shutil
import nn
from functools import reduce
from operator import mul
import logging

logger = logging.getLogger("default_log")


class TinySleepNet(object):

    def __init__(
            self,
            config,
            output_dir="./output",
            testing=False,
            use_best=False,
            use_rec=False,
            use_crb=False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = self.output_dir
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.use_rec = use_rec
        self.use_crb = use_crb
        self.pretrain = True

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.signals = tf.placeholder(dtype=tf.float32, shape=(None, 1, self.config["input_size"], 1),
                                          name='signals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='loss_weights')
            self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        net, self.rec_loss = self.build_cnn()
        net = self.append_rnn(net)
        net = nn.fc("softmax_linear", net, self.config["n_classes"], bias=0.0)
        # print(net.shape)
        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)
        # Cross-entropy loss
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )

        with tf.name_scope("loss_ce_mean") as scope:

            # Weight by sequence
            loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)

            # Weight by class
            sample_weights = tf.reduce_sum(
                tf.multiply(
                    tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
                    np.asarray(self.config["class_weights"], dtype=np.float32)
                ), 1
            )
            loss_w_class = tf.multiply(loss_w_seq, sample_weights)

            # Computer average loss scaled with the sequence length
            self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)
        # Regularization loss
        self.reg_losses = self.regularization_loss()
        # Total loss
        self.weight_rec = config["rec_weight"] if use_rec else 0
        self.loss = self.loss_ce + self.reg_losses
        # Metrics (used when we want to compute a metric from the output from minibatches)
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "loss_ce": tf.metrics.mean(values=self.loss_ce),
                "loss_reg": tf.metrics.mean(values=self.reg_losses),
                "loss_rec": tf.metrics.mean(values=self.rec_loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/loss_ce": self.loss_ce,
            "train/loss_rec": self.rec_loss,
            "train/loss_reg": self.reg_losses,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
            "train/init_state": self.init_state,
            "train/final_state": self.final_state,
        }
        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
            "test/init_state": self.init_state,
            "test/final_state": self.final_state,
        }
        # Tensoflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))

        # Optimizer
        if not testing:
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining

                    if (self.use_rec):
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss_ce + self.reg_losses + 1e-5 * self.rec_loss,
                            # loss=self.loss_ce + self.reg_losses,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,

                            clip_value=self.config["clip_grad_value"],
                        )
                    else:
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss_ce + self.reg_losses,
                            # loss=self.loss_ce + self.reg_losses,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,

                            clip_value=self.config["clip_grad_value"],
                        )
            self.pretrain_step_op, self.grad_op = nn.adam_optimizer(
                loss=self.rec_loss,
                training_variables=tf.trainable_variables(),
                global_step=self.global_step,
                # learning_rate=self.config["learning_rate"],
                learning_rate=self.lr,
            )
        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver([i for i in tf.global_variables()], max_to_keep=1)

        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            print(self.best_ckpt_path)
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    print("..............................")
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    # logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True
        if not is_restore:
            logger.info("Model started from random weights")

    def build_cnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)
        with tf.variable_scope("cnn") as scope:
            # layer0
            net = self.signals
            dim = 128
            # net = nn.conv1d("conv1d_0_1", net, int(max(2*self.w,4)), 7, 2*self.w+1)
            # net = nn.conv1d("conv1d_0_2", net, int(max(2 * self.w, 4)), 50, 1)
            # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_0")
            # layer1
            # print(net.shape)
            net = nn.conv1d("conv1d_1", net, dim, 50, 6)
            net = nn.max_pool1d("maxpool1d_1", net, 8, 8)
            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_1")
            # layer2
            net = nn.conv1d("conv1d_2_1", net, dim, 8, 1)
            fl = net

            net = nn.conv1d("conv1d_2_2", net, dim, 8, 1)
            net = nn.conv1d("conv1d_2_3", net, dim, 8, 1)
            rec = nn.conv1d("rec_2_1", net, dim, 8, 1)
            rec = nn.conv1d("rec_2_2", rec, dim, 8, 1)

            rec_loss = tf.reduce_mean(tf.pow(fl - rec, 2))
            net = nn.max_pool1d("maxpool1d_2", net, 4, 4)
            # layer3
            # net = nn.conv1d("conv1d_3_2", net, dim*2, 8, 1)
            # net = nn.conv1d("conv1d_3_3", net, dim*2, 8, 1)
            ave = nn.ave_pool1d("avepool1d", net, net.shape[2], net.shape[2])
            max = nn.max_pool1d("maxpool1d", net, net.shape[2], net.shape[2])
            net = tf.concat((ave, max), axis=1)
            # print(net.shape)
            # ("....")
            net = tf.layers.flatten(net, name="flatten_2", data_format="channels_first")
        net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_2")
        # print(net.shape)

        return net, rec_loss

    def append_rnn(self, inputs):
        unit_num = (1) * self.config["n_rnn_units"]
        # print(inputs.shape)
        with tf.variable_scope("rnn") as scope:
            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            # seq_inputs = nn.sa(seq_inputs,16, dropout = 0.2, layer_num=4)
            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(unit_num))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )

            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, unit_num], name="reshape_nonseq_input")

            # net = tf.layers.dropout(net, rate=0.75, training=self.is_training, name="drop")
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        print(num_params)
        return net

    def train(self, minibatches):
        self.run(self.metric_init_op)
        preds = []
        trues = []
        losses = []
        for x, y, w, sl, re in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: True,
                self.loss_weights: w,
                self.seq_lengths: sl,
            }

            if re:
                # Initialize state of RNN
                state = self.run(self.init_state)

            # Carry the states from the previous batches through time
            for i, (c, h) in enumerate(self.init_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            if self.use_rec:
                if (self.pretrain):
                    _, outputs = self.run([self.pretrain_step_op, self.train_outputs], feed_dict=feed_dict)
                else:
                    _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
            else:
                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
            # Buffer the final states
            state = outputs["train/final_state"]
            losses.append(outputs["train/loss"])
            tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        loss = np.mean(losses)
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        return acc,f1_score,loss

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []


        for x, y, w, sl, re in minibatches:
            feed_dict = {
                self.signals: x,
                self.labels: y,
                self.is_training: False,
                self.loss_weights: w,
                self.seq_lengths: sl,
            }

            if re:
                # Initialize state of RNN
                state = self.run(self.init_state)

            # Carry the states from the previous batches through time
            for i, (c, h) in enumerate(self.init_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            outputs = self.run(self.test_outputs, feed_dict=feed_dict)

            # Buffer the final states
            state = outputs["test/final_state"]

            losses.append(outputs["test/loss"])

            tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y, (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return acc,f1_score,cm


    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)

    def save_checkpoint(self, name):
        path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved checkpoint to {}".format(path))

    def save_best_checkpoint(self, name):
        if not os.path.exists(self.best_ckpt_path):
            os.makedirs(self.best_ckpt_path)
        path = self.best_saver.save(
            self.sess,
            os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )
        logger.info("Saved best checkpoint to {}".format(path))

    def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Save weights
        path = os.path.join(self.weights_path, "{}.npz".format(name))
        logger.info("Saving weights in scope: {} to {}".format(scope, path))
        save_dict = {}
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        for v in cnn_vars:
            save_dict[v.name] = self.sess.run(v)
            logger.info("  variable: {}".format(v.name))
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        np.savez(path, **save_dict)

    def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Load weights
        logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        with np.load(weight_file) as f:
            for v in cnn_vars:
                tensor = tf.get_default_graph().get_tensor_by_name(v.name)
                self.run(tf.assign(tensor, f[v.name]))
                logger.info("  variable: {}".format(v.name))

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "cnn/conv1d_1/conv2d/kernel:0",
            "cnn/conv1d_2_1/conv2d/kernel:0",
            "cnn/conv1d_2_2/conv2d/kernel:0",
            "cnn/conv1d_2_3/conv2d/kernel:0",
            # "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses
