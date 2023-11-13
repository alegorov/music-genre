import random

import numpy as np
import pandas as pd

NUM_PARTS = 10
NUM_TAGS = 256


def save_csv(file_path, header, data):
    with open(file_path, 'w') as f:
        f.write(header + '\n')
        for s in data:
            f.write(s + '\n')

    target = np.zeros(NUM_TAGS, dtype=np.int32)

    df = pd.read_csv(file_path)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        tags = [int(x) for x in row.tags.split(',')]
        target[tags] = 1

    return np.min(target)



def main():
    random.seed(42)

    with open('../data/train.csv') as f:
        src = [s for s in map(lambda line: line.strip(), f) if s]
    header = src.pop(0)
    random.shuffle(src)

    parts = [[] for _ in range(NUM_PARTS)]
    for i, s in enumerate(src):
        parts[i % NUM_PARTS].append(s)

    for split_id in range(NUM_PARTS):
        train_data = []
        val_data = []
        for i in range(NUM_PARTS):
            if i == split_id:
                val_data = parts[i]
            else:
                train_data += parts[i]
        random.shuffle(train_data)

        train_status = save_csv('../data/split_train{}.csv'.format(split_id), header, train_data)
        val_status = save_csv('../data/split_val{}.csv'.format(split_id), header, val_data)

        print('split_id: {}, train_status: {}, val_status: {}'.format(split_id, train_status, val_status))


if __name__ == '__main__':
    main()
