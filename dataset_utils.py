

import json
from meta import *
import os
import cv2

import numpy as np

'''
block = {"entry_number": entry_number, "entries": entries}
entries = [entry]
entry = float[32768..0]
'''

def get_block_names(option):
    # train /test

    dirlist = os.listdir(dataset)
    i = 0
    while i < len(dirlist):
        if "block" not in dirlist[i] or ".json" not in dirlist[i]: del dirlist[i]
        else: i += 1
    file_nums = len(dirlist)
    section = int(np.ceil(.8 * file_nums))


    if option == 'train':  return dirlist[:section]

    elif option == 'test': return dirlist[section:]

    mix_list = os.listdir(dataset)
    i = 0
    while i < len(dirlist):
        if "mix" not in dirlist[i] or ".json" not in dirlist[i]: del dirlist[i]
        else: i += 1
    return mix_list


def get_entry_num(blockname):
    block_dict = json.load(open(blockname, "r"))
    return block_dict['entry_number']


def load_trivial(blk):
    print(blk)
    block = json.load( open(dataset+blk, "r") )

    total = block['entry_number']
    specs = np.array(block['entries'][0]).reshape(total, 128, 128)
    attens = np.array(block['entries'][1]).reshape(total, 2)

    return specs, attens


def load_denoise(blk):
    pass
    # block = json.load( open(blk, "r") )

    # total = block['entry_number']
    # mixed_specs = np.array(block['entries'][0]).reshape(total, 256, 64)
    # attens = np.array(block['entries'][1]).reshape(total, 2)
    # clean_specs = np.array(block['entries'][1]).reshape(total, 256, 64)

    # return mixed_specs, attens, clean_specs


# if __name__=="__main__":
#     for blk in get_block_names('train'):
#         specs, _ = load_trivial(blk)
#         for i in range(len(specs)):
#             cv2.imwrite(dataset + blk[:-5] + str(i) + ".png", specs[i].reshape(128, 64) * 255)

