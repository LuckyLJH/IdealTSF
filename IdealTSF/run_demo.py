import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from IdealTSF import TimeSeriesAugmentations
# from IdealTSF import IdealTSFArchitecture
from IdealTSF import Pretrainer, Trainer


def set_global_seed(seed: int = 2025):
    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch（CPU + CUDA）
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_ETTh1_dataset(seq_len, pred_len, time_increment=1):
    file_name = ".\\dataset\\ETTh1.csv"
    df_raw = pd.read_csv(file_name, index_col=0)
    n = len(df_raw)
    train_end = 12 * 30 * 24
    val_end = train_end + 4 * 30 * 24
    test_end = val_end + 4 * 30 * 24

    train_df = df_raw[:train_end]
    val_df = df_raw[train_end - seq_len: val_end]
    test_df = df_raw[val_end - seq_len: test_end]

    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

    x_train, y_train = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment)
    x_val, y_val = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment)
    x_test, y_test = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment)

    flatten = lambda y: y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    return (x_train, flatten(y_train)), (x_val, flatten(y_val)), (x_test, flatten(y_test))


def construct_sliding_window_data(data, seq_len, pred_len, time_increment=1):
    n_samples = data.shape[0] - (seq_len - 1) - pred_len
    range_ = np.arange(0, n_samples, time_increment)
    x, y = list(), list()
    for i in range_:
        x.append(data[i:(i + seq_len)].T)
        y.append(data[(i + seq_len):(i + seq_len + pred_len)].T)
    return np.array(x), np.array(y)


if __name__ == '__main__':

    # SEED = None
    # set_global_seed(SEED)

    # ----- Step 1: Load Data -----
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_ETTh1_dataset(seq_len=512, pred_len=96)

    # ----- Step 2: Define augmentations for pretraining -----
    augmentations = TimeSeriesAugmentations(
        noise_std=0.01,
        erase_prob=0.3,
        noise_scales=[0.1, 0.05, 0.02, 0.01]
    )

    # ----- Step 3: Pretraining -----
    pretrainer = Pretrainer(
        device='cuda:0',
        num_epochs=30,
        batch_size=32,
        learning_rate=5e-3,
        # 5e-3
        weight_decay=1e-4,
        rho=0.9,
        augmentations=augmentations,
        use_revin=True
    )
    pretrained_model = pretrainer.pretrain(x_train, y_train)

    # ----- Step 4: Fine-tuning -----
    trainer = Trainer(
        pretrained_model=pretrained_model,
        device='cuda:0',
        num_epochs=30,
        batch_size=64,
        # 32 32 128
        # 128
        learning_rate=5e-4,
        # 5e-4 4e-4
        weight_decay=1e-4,
        rho=0.9,
        use_revin=True
    )
    trainer.fine_tune(x_train, y_train)

    # ----- Step 5: Evaluation -----
    y_pred_test, attention_test = trainer.forecast(x_test)

    mse = np.mean((y_test - y_pred_test) ** 2)
    mae = np.mean(np.abs(y_test - y_pred_test))
    print('MSE:', mse)
    print('MAE:', mae)
