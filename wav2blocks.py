

from meta import *
import dataset_utils
from dataset_utils import *


import numpy as np

import scipy
import scipy.io.wavfile

import cv2

import matplotlib.pyplot as plt
import gc

import random


import os


BLOCK_THRES = 150
FREQ = 512          # cut to 128
DUR = 128
SEG_STEP = 64
MAX_BLOCKS = 100


def _squeezec(spec):
    newchan = int(spec.shape[0]//2)
    clist = np.arange(newchan)
    newspec = (spec[2*clist] + spec[2*clist+1]) / 2
    return newspec

def _squeeze(spec):
    newchan = int(spec.shape[0]//2)
    newlen  = int(spec.shape[1]//2)
    clist = np.arange(newchan)
    llist = np.arange(newlen)
    newspec = (spec[2*clist] + spec[2*clist+1]) / 2
    newspec = (newspec[:,2*llist] + newspec[:,2*llist+1]) / 2
    return newspec

def _wav2arrays(wav):
    # spec of all
    fs, x = scipy.io.wavfile.read(wav)
    spec, _, _, _= plt.specgram(x, Fs=fs, NFFT=2048, noverlap=1900); plt.close('all'); gc.collect()

    # pre - cut/modify
    # FREQ //= 4
    spec = spec[:FREQ]
    spec[spec > 255] = 255; spec /= 255
    spec = _squeezec(spec)[:FREQ//4]

    # specs
    specs = []
    num_segs = 1 + (spec.shape[1] - DUR)//SEG_STEP
    real_segs = 0
    for i in range(num_segs):
        new = spec[:, i*SEG_STEP:i*SEG_STEP+DUR]
        if np.sum(new) > 500:
            real_segs += 1
            specs.append(new.reshape( int(FREQ*DUR/4) ))

    # atten
    if 'B' in wav: tp = 1
    else:          tp = 0

    attens = np.zeros( (real_segs,2), dtype=float)
    attens[:,tp] = 1
    attens = list(attens)

    return specs, attens


_have_both = []


def createBlocks():
    '''
    blocks have names: blockXX.json, having 200 items in each
    '''
    block_count = 0
    block_specs = []
    block_attens = []

    os.chdir(wav_clip_root)
    wav_clips = os.listdir()
    r = 0
    while r < len(wav_clips):
        if '.wav' not in wav_clips[r] or '.png' in wav_clips[r]: del wav_clips[r]
        else: r += 1
    os.chdir(proj_root)

    random.shuffle(wav_clips)
    for wav in wav_clips:
        if block_count > MAX_BLOCKS: break

        print("\t", wav)
        specs, attens = _wav2arrays(wav_clip_root + wav)
        num_entries = len(specs)
        
        if len(block_specs) + num_entries > BLOCK_THRES:

            blockname = "block" + str(block_count) + ".json"
            jh = open(dataset+blockname, "w")
            

            random.shuffle(block_specs)
            block_specs = np.array(block_specs).tolist()

            random.shuffle(block_attens)
            block_attens = np.array(block_attens).tolist()
            
            a_indexes = []
            b_indexes = []
            for i in range(len(block_specs)):
                if block_attens[i][0] == 1: a_indexes.append(i)
                else: b_indexes.append(i)
            if len(a_indexes) > 0 and len(b_indexes) > 0:
                # print("probe")
                _have_both.append(blockname)

            print("\tready to dump")
            json_dict = { \
                "entry_number":len(block_specs), \
                "entries": (block_specs, block_attens), \
                "a_indexes": a_indexes, \
                "b_indexes": b_indexes,
            }
            json.dump(json_dict, jh)
            jh.close()
            print("\tdump done")


            block_count += 1
            print( "blockcount = {} with {} entries".format(block_count, len(block_specs)) )
            block_specs = []
            block_attens = []

        block_specs.extend(specs)
        block_attens.extend(attens)


def mix():
    test_blocks = dataset_utils.get_block_names('test')
    print(test_blocks)
    print(_have_both)

    mixed_count = 0
    for blk in test_blocks:
        
        # if blk not in _have_both: continue

        print(blk)
        block = json.load(open(dataset+blk, "r"))
        # one blockX.json, one mix.json
        specs = np.array(block['entries'][0])
        attens = np.array(block['entries'][1])
        a_indexes = block['a_indexes']
        b_indexes = block['b_indexes']

        if len(a_indexes)==0 or len(b_indexes)==0: continue

        # (mixed, atten, pure) [|block|-1..0]
        mixed_list = []
        atten_list = []
        pure_list  = []
        b_index_run = 0
        a_b = 0
        for a_index_run in range(len(a_indexes)):
            # a_indexes[a_index_run], b_indexes[b_index_run]
            mixed = specs[a_indexes[a_index_run]] + specs[b_indexes[b_index_run]]
            mixed[mixed > 1] = 1
            atten = np.zeros(2, dtype=float); atten[a_b] = 1
            if a_b == 0:
                pure = specs[a_indexes[a_index_run]]
            else:
                pure = specs[b_indexes[b_index_run]]

            mixed_list.append(mixed)
            atten_list.append(atten)
            pure_list.append(pure)

            b_index_run += 1
            if b_index_run >= len(b_indexes):
                b_index_run = 0

        mixed_list = np.array(mixed_list).tolist()
        atten_list = np.array(atten_list).tolist()
        pure_list  = np.array(pure_list).tolist()

        mixed_dict = {"entry_number": len(atten_list), "entries": [mixed_list, atten_list, pure_list]}
        jh = open(dataset+"mixed"+str(mixed_count)+".json", "w")
        json.dump(mixed_dict, jh)

        mixed_count += 1

