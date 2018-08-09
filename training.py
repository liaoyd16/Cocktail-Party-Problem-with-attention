# training.py
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import cv2

import model.fb_model as model
from model.fb_model import ResDAE

import pickle
import logger
import dataset_utils
from meta import *

import os
import pdb
from noise import white



INF = 100
DECAY = 0.75


if __name__=="__main__":

    # iteration / observation meta params
    ITER = 6000
    BS = 5
    OBS = 1


    # directory & index control
    block_names = dataset_utils.get_block_names("train")
    block_iter = 0
    temp_specs, temp_attens = dataset_utils.load_trivial(block_names[0])
    block_pos = 0

    clear_dir(logroot, kw="events")


    # training prep
    model = ResDAE() # model = pickle.load(ph)
    lr = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9); optimizer.zero_grad()
    lossF = nn.MSELoss(size_average=True) # must be True


    # training
    logger = logger.Logger(logroot)
    epoch_count = 0
    for t in range(ITER):

        ''' indexing '''
        end = min(block_pos+BS, len(temp_specs))


        ''' load batch '''
        source = temp_specs[block_pos:end]
        source = torch.tensor(source, dtype=torch.float)
        # source = source + white(source)

        target = temp_attens[block_pos:end]
        target = torch.tensor(target, dtype=torch.float)


        ''' forward pass'''
        top = model.upward(source + white(source))
        recover = model.downward(top).view(end-block_pos, 128, 128)


        ''' loss-bp '''
        loss = lossF(recover, source)
        loss.backward()


        ''' observe '''
        logger.scalar_summary("loss", loss, t)
        print("\nt={} loss={}".format(t, loss) )

        if t%OBS==0:
            if OBS < 32: OBS *= 2

            # os.chdir(training_result)
            spec = source[0].view(128, 128).detach().numpy()
            cv2.imwrite(training_result+"source_{}.png".format(t), spec * 255)

            y1 = recover[0].view(128, 128).detach().numpy()
            cv2.imwrite(training_result+"y1_{}.png".format(t), y1 * 255)


        ''' stepping '''
        optimizer.step()
        optimizer.zero_grad()


        ''' next batch '''
        block_pos += BS
        if block_pos >= len(temp_attens):
            block_iter += 1
            if block_iter >= len(block_names): block_iter = 0
            temp_specs, temp_attens = dataset_utils.load_trivial(block_names[block_iter])

            block_pos = 0


    ''' dump '''
    ph = open("dae.pickle", "wb")
    pickle.dump(model, ph)

#end