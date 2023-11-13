from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import AttentionPooling
from utils import FeedForward

BEST_MODEL_NAME = 'best3'

BATCH_SIZE = 48
NUM_EPOCHS = 20
LR = 1e-4

CUDA_DEV = 'cuda'
NUM_TAGS = 256


class TaggingDataset(Dataset):
    def __init__(self, base_dir, df, randomize, testing):
        self.base_dir = base_dir
        self.df = df
        self.randomize = randomize
        self.testing = testing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track
        embeds = np.load('{}/data/track_embeddings/{}.npy'.format(self.base_dir, track_idx))

        repeats = (420 + embeds.shape[0] - 1) // embeds.shape[0]
        embeds = np.concatenate([embeds] * repeats, axis=0)[:420]

        if self.randomize:
            indices = np.random.randint(0, embeds.shape[0], embeds.shape[0])
            embeds = embeds[indices, :]

        if self.testing:
            return track_idx, embeds
        tags = [int(x) for x in row.tags.split(',')]
        target = np.zeros(NUM_TAGS, dtype=np.float32)
        target[tags] = 1
        return track_idx, embeds, target


class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_features_size = 512

        input_size = 768
        embedding_size = 512

        self.proj = FeedForward(input_size, embedding_size, mult=4)
        self.conformer = torchaudio.models.Conformer(
            input_dim=embedding_size,
            num_heads=4,
            ffn_dim=512,
            num_layers=3,
            depthwise_conv_kernel_size=31
        )
        self.pooling = AttentionPooling(self.output_features_size)

    def forward(self, x):
        lengths = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.int64, device=CUDA_DEV)

        x = self.proj(x)
        x = self.conformer(x, lengths)
        x = x[0]
        x = self.pooling(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_net = BasicNet()

        hidden_dim = self.basic_net.output_features_size

        self.bn = nn.LayerNorm(hidden_dim)

        self.mult_coef = 512
        self.mult = nn.Linear(hidden_dim, NUM_TAGS * self.mult_coef)
        self.out = nn.Sequential(
            nn.LayerNorm(self.mult_coef),
            nn.ReLU(),
            nn.Linear(self.mult_coef, 1),
        )

    def forward(self, x):
        x = self.basic_net(x)
        x = self.bn(x)
        x = self.mult(x)
        x = torch.reshape(x, (-1, self.mult_coef))
        x = self.out(x)
        x = torch.reshape(x, (-1, NUM_TAGS))
        return x


def train_epoch(model, loader, criterion, optimizer):
    print(' start...', flush=True)
    model.train()
    running_loss = None
    alpha = 0.8
    for iteration, data in enumerate(loader):
        optimizer.zero_grad()
        track_idxs, embeds, target = data
        pred_logits = model(embeds)
        ce_loss = criterion(pred_logits, target)
        ce_loss.backward()
        optimizer.step()

        if running_loss is None:
            running_loss = ce_loss.item()
        else:
            running_loss = alpha * running_loss + (1 - alpha) * ce_loss.item()
        if iteration % 100 == 0:
            print('   {} batch {} loss {}'.format(
                datetime.now(), iteration + 1, running_loss
            ), flush=True)


def get_precision(model, loader):
    model.eval()
    y_true = []
    y_scores = []
    with torch.no_grad():
        for data in loader:
            track_idxs, embeds, target = data
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)

            y_true.append(target.cpu().numpy())
            y_scores.append(pred_probs.cpu().numpy())

    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)

    return average_precision_score(y_true, y_scores)


def predict(model, loader):
    model.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            track_idx, embeds = data
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)
            pred_probs = pred_probs.cpu().numpy()

            for i, track_id in enumerate(track_idx):
                track_idxs.append(track_id)
                predictions.append(pred_probs[i])

    return track_idxs, predictions


def collate_fn(b):
    track_idxs = [x[0] for x in b]
    embeds = torch.tensor(np.array([x[1] for x in b], dtype=np.float32), device=CUDA_DEV)

    if len(b[0]) == 2:
        return track_idxs, embeds

    targets = torch.tensor(np.array([x[2] for x in b], dtype=np.float32), device=CUDA_DEV)
    return track_idxs, embeds, targets


def train(base_dir, train_dataloader, val_dataloader):
    model = Network().to(CUDA_DEV)
    criterion = nn.BCEWithLogitsLoss().to(CUDA_DEV)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    max_precision = 0.
    best_epoch = 0
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_epoch(model, train_dataloader, criterion, optimizer)
        precision = get_precision(model, val_dataloader)
        if precision > max_precision:
            max_precision = precision
            best_epoch = epoch + 1
            checkpoint_path = '{}/{}.pt'.format(base_dir, BEST_MODEL_NAME)
            torch.save(model.state_dict(), checkpoint_path)
            with open(checkpoint_path + '.txt', 'w') as text_file:
                print('Validation on epoch {}'.format(epoch + 1), file=text_file)
                print('Precision: {}'.format(precision), file=text_file)
        print('Validation on epoch ({})'.format(epoch + 1), flush=True)
        print('Precision: {}, Best: {} ({})'.format(precision, max_precision, best_epoch), flush=True)


def main():
    parser = ArgumentParser(description='Simple naive baseline')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=True)
    parser.add_argument('--is-train', dest='is_train', action='store', required=True)
    parser.add_argument('--split-id', dest='split_id', action='store', required=True)
    args = parser.parse_args()

    global BEST_MODEL_NAME
    BEST_MODEL_NAME += '-' + args.split_id

    if int(args.is_train):
        print('Train')

        df_train = pd.read_csv('{}/data/split_train{}.csv'.format(args.base_dir, args.split_id))
        df_val = pd.read_csv('{}/data/split_val{}.csv'.format(args.base_dir, args.split_id))

        train_dataset = TaggingDataset(args.base_dir, df_train, True, False)
        val_dataset = TaggingDataset(args.base_dir, df_val, False, False)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        train(args.base_dir, train_dataloader, val_dataloader)
    else:
        print('Submission')

        df_test = pd.read_csv(args.base_dir + '/data/test.csv')
        test_dataset = TaggingDataset(args.base_dir, df_test, False, True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = Network().to(CUDA_DEV)
        model.load_state_dict(torch.load('{}/{}.pt'.format(args.base_dir, BEST_MODEL_NAME)))

        track_idxs, predictions = predict(model, test_dataloader)

        predictions_df = pd.DataFrame([
            {'track': track, 'prediction': ','.join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ])

        predictions_df.to_csv('{}/prediction-{}.csv'.format(args.base_dir, BEST_MODEL_NAME), index=False)


if __name__ == '__main__':
    main()
