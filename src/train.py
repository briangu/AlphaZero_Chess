#!/usr/bin/env python

from alpha_net import ChessNet, train, board_data
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def train_chessnet(net_to_train="current_net_trained7_iter1.pth.tar",save_as="current_net_trained8_iter1.pth.tar"):
    # gather data
    data_path = "./datasets/iter1/"
    datasets = []
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    data_path = "./datasets/iter0/"
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    datasets = np.array(datasets)

    # train net
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])

    train_set = board_data(datasets)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)

    train(net,train_loader)
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))

if __name__=="__main__":
    train_chessnet()