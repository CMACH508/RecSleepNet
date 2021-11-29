config = {
    # Train
    "pretrain_epochs": 10,
    "n_epochs": 100,
    "learning_rate": 1e-4,
    "clip_grad_value": 5.0,
    "seq_length": 20,
    "batch_size": 15,
    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "rec_weight":1e-5,
    "class_weights":[1,1.5,1,1,1],
    # Dataset
    "dataset": "",
    "suffix":"",
    # Fold number
    "sleepedf":20,
    "sleepedfx":10,
    "isruc":10,
    "ucddb":25,
}


