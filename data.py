import os
import re

import numpy as np


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid}.npz"
    elif "ucddb" in dataset:
        reg_exp = f"ucddb{str(sid).zfill(3)}.npz"
    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
    #print(len(subject_files))
    return subject_files


def load_data(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []

    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels
import glob
def load_seq_ids(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    with open(fname, "r") as f:
        for line in f:
            ids.append(int(line.strip()))
    ids = np.asarray(ids)
    return ids
def load_by_id(dataset,subject_files,sids):
    files = []
    for sid in sids:
        files.append(get_subject_files(
            dataset=dataset,
            files=subject_files,
            sid=sid,
        ))
    x, y = load_data(np.hstack(files))
    return x,y
def get_split(config,fold_idx,random_seed=42):
    dataset=config["dataset"]
    subject_files = glob.glob(os.path.join("./data",dataset, "*.npz"))
    # Load subject IDs
    fname = "{}.txt".format(dataset)
    seq_sids = load_seq_ids(fname)
    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config[dataset])

    test_sids = fold_pids[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)

    # Further split training set as validation set (10%)
    n_valids = round(len(train_sids) * 0.10)

    # Set random seed to control the randomness
    np.random.seed(random_seed)
    valid_sids = np.random.choice(train_sids, size=n_valids, replace=False)
    train_sids = np.setdiff1d(train_sids, valid_sids)

    print("Train SIDs: ({}) {}".format(len(train_sids), train_sids))
    print("Valid SIDs: ({}) {}".format(len(valid_sids), valid_sids))
    print("Test SIDs: ({}) {}".format(len(test_sids), test_sids))
    train_x, train_y = load_by_id(dataset,subject_files,train_sids)
    valid_x, valid_y = load_by_id(dataset,subject_files,valid_sids)
    test_x, test_y = load_by_id(dataset, subject_files, test_sids)
    return train_x,train_y,valid_x,valid_y,test_x,test_y