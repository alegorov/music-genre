import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

WEIGHTS = [1., 0.67, 0.9, 0.79, 0.67]

NUM_TAGS = 256


def get_probs(data, w, track_id, data_id):
    x = np.vstack([d[data_id][track_id][np.newaxis, :] for d in data])
    x = np.sum(w * x, axis=0)
    x = torch.sigmoid(torch.tensor(x))
    return x.numpy()


def train(data, w, df):
    y_true = []
    y_scores = []

    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        track_id = row.track

        probs = get_probs(data, w, track_id, 0)
        probs = probs[np.newaxis, :]
        y_scores.append(probs)

        tags = [int(x) for x in row.tags.split(',')]
        target = np.zeros((1, NUM_TAGS), dtype=np.float32)
        target[0, tags] = 1
        y_true.append(target)

    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)

    print(average_precision_score(y_true, y_scores))


def predict(data, w, df, file_path):
    track_ids = []
    predictions = []

    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        track_id = row.track

        probs = get_probs(data, w, track_id, 1)

        track_ids.append(track_id)
        predictions.append(probs)

    predictions_df = pd.DataFrame([
        {'track': track, 'prediction': ','.join([str(p) for p in probs])}
        for track, probs in zip(track_ids, predictions)
    ])

    predictions_df.to_csv(file_path, index=False)


def main():
    parser = ArgumentParser(description='Ensemble')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=True)
    parser.add_argument('--is-train', dest='is_train', action='store', required=True)
    args = parser.parse_args()

    data = [None] * len(WEIGHTS)
    for algorithm_id in range(1, len(WEIGHTS) + 1):
        with open('{}/data{}.pkl'.format(args.base_dir, algorithm_id), 'rb') as f:
            data[algorithm_id - 1] = pickle.load(f)

    w = np.array(WEIGHTS, dtype=np.float32)
    w /= w.sum()
    w = w[:, np.newaxis]

    if int(args.is_train):
        df = pd.read_csv(args.base_dir + '/data/train.csv')
        train(data, w, df)
    else:
        df = pd.read_csv(args.base_dir + '/data/test.csv')
        file_path = args.base_dir + '/prediction.csv'
        predict(data, w, df, file_path)


if __name__ == '__main__':
    main()
