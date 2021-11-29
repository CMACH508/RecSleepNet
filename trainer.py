import argparse
import importlib
import os
import tensorflow as tf
from train import train
from config import config
def run(dataset, gpu, from_fold, to_fold, suffix='', random_seed=42):
    # Set GPU visible to Tensorflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output_dir = dataset+"_"+suffix
    config["dataset"] = dataset
    assert from_fold<=to_fold
    assert to_fold<config[dataset]
    for fold_idx in range(from_fold, to_fold+1):
        train(
            config=config,
            fold_idx=fold_idx,
            output_dir=os.path.join(output_dir, 'train'),
            log_file=os.path.join(output_dir, f'train_{gpu}.log'),
            restart = True,
            random_seed=random_seed+fold_idx,
        )
        tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--from_fold", type=int, required=True)
    parser.add_argument("--to_fold", type=int, required=True)
    parser.add_argument("--suffix", type=str, default='')
    args = parser.parse_args()
    run(
        dataset=args.db,
        gpu=args.gpu,
        from_fold=args.from_fold,
        to_fold=args.to_fold,
        suffix=args.suffix,
    )

