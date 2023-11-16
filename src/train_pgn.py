import chess
import chess.pgn
import io
import torch
from torch.utils.data import Dataset
import encoder_decoder as ed
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import sys
from alpha_net import ChessNet, AlphaLoss
import os
from chess_board import board as c_board
from torch.utils.data import IterableDataset
import numpy as np
from collections import deque
import time

from util_pgn import *


def train(net, train_loader, out_model_path, epoch_start=0, epoch_stop=20, cpu=0, batch_size=128):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1000, threshold=0.01)

    torch.save({'state_dict': net.state_dict()}, os.path.join(out_model_path, "epoch_start.pth.tar"))

    # losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        # losses_per_batch = deque(maxlen=100)
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f LR: %f' %
                      (os.getpid(), epoch + 1, (i + 1)*batch_size, len(train_loader.dataset), total_loss/10, optimizer.param_groups[0]['lr']))
                print("Policy:", policy[0].argmax().item(), policy_pred[0].argmax().item(), "Value:", value[0].item(), value_pred[0,0].item())
                # losses_per_batch.append(total_loss/10)
                total_loss = 0.0
                # avg_loss = sum(losses_per_batch) / len(losses_per_batch)
                # scheduler.step(avg_loss)
            if (i+1) % 100000 == 0:
                torch.save({'state_dict': net.state_dict()}, os.path.join(out_model_path, f"epoch_{epoch}_{i}.pth.tar"))
            if (i+1) % 10000 == 0:
                scheduler.step()

        torch.save({'state_dict': net.state_dict()}, os.path.join(out_model_path, f"epoch_{epoch}.pth.tar"))
        # losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        # if len(losses_per_epoch) > 100:
        #     if abs(sum(losses_per_epoch[-4:-1])/3 - sum(losses_per_epoch[-16:-13])/3) <= 0.01:
        #         break
        # scheduler.step()


def train_chessnet(train_loader, net_to_train, out_model_path, batch_size=128, last_n_moves=8):
    net = ChessNet(last_n_moves=last_n_moves)
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    if net_to_train is not None:
        checkpoint = torch.load(net_to_train)
        net.load_state_dict(checkpoint['state_dict'])
    train(net,train_loader, out_model_path, batch_size=batch_size)
    # torch.save({'state_dict': net.state_dict()}, save_as)


if __name__=="__main__":
    out_model_path = sys.argv[1]
    out_model_path = os.path.join(out_model_path, str(time.time()))
    os.makedirs(out_model_path, exist_ok=True)
#    pgn_path = sys.argv[2] if len(sys.argv) > 2 else "/data/lichess_db_standard_rated_2023-02.pgn"
    pgn_path = "/data/lichess_db_standard_rated_2023-02.pgn"
    game_cnt = 108201825
    # model_path = sys.argv[3] if len(sys.argv) > 3 else None
    last_n_moves = 8
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    dataset = ChessPGNDataset(pgn_path, game_cnt, last_n_moves)
    batch_size = 128  # You can adjust the batch size as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    train_chessnet(train_loader, net_to_train=model_path,out_model_path=out_model_path, batch_size=batch_size)
