import pickle
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader

from train1 import Network as Network1
from train1 import TaggingDataset as TaggingDataset1
from train1 import collate_fn as collate_fn1
from train2 import Network as Network2
from train2 import TaggingDataset as TaggingDataset2
from train2 import collate_fn as collate_fn2
from train3 import Network as Network3
from train3 import TaggingDataset as TaggingDataset3
from train3 import collate_fn as collate_fn3
from train4 import Network as Network4
from train4 import TaggingDataset as TaggingDataset4
from train4 import collate_fn as collate_fn4
from train5 import Network as Network5
from train5 import TaggingDataset as TaggingDataset5
from train5 import collate_fn as collate_fn5

NUM_PARTS = 10
BATCH_SIZE = 32
CUDA_DEV = 'cuda'


def predict(model, loader, testing):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for data in loader:
            track_idx = data[0]
            data = data[1:]
            if not testing:
                data = data[:-1]
            pred_logits = model(*data)
            pred_logits = pred_logits.cpu().numpy()

            for i, track_id in enumerate(track_idx):
                predictions[track_id] = pred_logits[i]
    return predictions


def main():
    parser = ArgumentParser(description='Predict')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=True)
    parser.add_argument('--algorithm-id', dest='algorithm_id', action='store', required=True)
    args = parser.parse_args()

    TaggingDataset = None
    if args.algorithm_id == '1':
        TaggingDataset = TaggingDataset1
    if args.algorithm_id == '2':
        TaggingDataset = TaggingDataset2
    if args.algorithm_id == '3':
        TaggingDataset = TaggingDataset3
    if args.algorithm_id == '4':
        TaggingDataset = TaggingDataset4
    if args.algorithm_id == '5':
        TaggingDataset = TaggingDataset5

    collate_fn = None
    if args.algorithm_id == '1':
        collate_fn = collate_fn1
    if args.algorithm_id == '2':
        collate_fn = collate_fn2
    if args.algorithm_id == '3':
        collate_fn = collate_fn3
    if args.algorithm_id == '4':
        collate_fn = collate_fn4
    if args.algorithm_id == '5':
        collate_fn = collate_fn5

    Network = None
    if args.algorithm_id == '1':
        Network = Network1
    if args.algorithm_id == '2':
        Network = Network2
    if args.algorithm_id == '3':
        Network = Network3
    if args.algorithm_id == '4':
        Network = Network4
    if args.algorithm_id == '5':
        Network = Network5

    df_val = [pd.read_csv('{}/data/split_val{}.csv'.format(args.base_dir, split_id)) for split_id in range(NUM_PARTS)]

    val_dataset = [TaggingDataset(args.base_dir, df_val[split_id], False, False) for split_id in range(NUM_PARTS)]

    val_dataloader = [DataLoader(val_dataset[split_id], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) for
                      split_id in range(NUM_PARTS)]

    df_test = pd.read_csv(args.base_dir + '/data/test.csv')
    test_dataset = TaggingDataset(args.base_dir, df_test, False, True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    val_data = {}
    test_data = {}

    for split_id in range(NUM_PARTS):
        model = Network().to(CUDA_DEV)
        model.load_state_dict(torch.load('{}/best{}-{}.pt'.format(args.base_dir, args.algorithm_id, split_id)))

        val_predict = predict(model, val_dataloader[split_id], False)
        test_predict = predict(model, test_dataloader, True)

        for k, v in val_predict.items():
            val_data[k] = v

        for k, v in test_predict.items():
            curr = test_data.get(k, None)
            if curr is None:
                test_data[k] = v
            else:
                test_data[k] = curr + v

    for k, v in test_data.items():
        test_data[k] = v / NUM_PARTS

    data = (val_data, test_data)
    with open('{}/data{}.pkl'.format(args.base_dir, args.algorithm_id), 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
