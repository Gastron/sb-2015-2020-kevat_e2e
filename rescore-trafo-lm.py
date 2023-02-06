#!/usr/bin/env python3

import os
import sys
import time
import tqdm
import torch
import logging
import speechbrain as sb
import itertools
from hyperpyyaml import load_hyperpyyaml
from types import SimpleNamespace
import pathlib

def setup(hparams, run_opts):
    """ Kind of mimics what Brain does """
    if "device" in run_opts:
        device = run_opts["device"]
    elif "device" in hparams:
        device = hparams["device"]
    else:
        device = "cpu"
    print("Device is:", device)
    if "cuda" in device:
        torch.cuda.set_device(int(device[-1]))
    modules = torch.nn.ModuleDict(hparams["modules"]).to(device)
    hparams = SimpleNamespace(**hparams)
    if hasattr(hparams, "checkpointer"):
        if hasattr(hparams, "test_max_key"):
            ckpt = hparams.checkpointer.find_checkpoint(max_key=hparams.test_max_key)
        elif hasattr(hparams, "test_min_key"):
            ckpt = hparams.checkpointer.find_checkpoint(min_key=hparams.test_min_key)
        else:
            ckpt = hparams.checkpointer.find_checkpoint()
        hparams.checkpointer.load_checkpoint(ckpt)
        epoch = hparams.epoch_counter.current
        print("Loaded checkpoint from epoch", epoch, "at path", ckpt.path)
    modules.eval()
    return modules, hparams, device

def count_lines(infile):
    lines = 0
    with open(infile) as fin:
        for _ in fin:
            lines += 1
    return lines

def text_io(infile):
    with open(infile) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            yield uttid, text



def run_test(modules, hparams, device):
    testfile = pathlib.Path(hparams.testfile)
    num_utts = count_lines(testfile)
    data_iter = text_io(testfile)
    bosl = [hparams.bos_index]
    eosl = [hparams.eos_index]
    with open(hparams.test_out, 'w') as fo:
        with torch.no_grad():
            for uttid, text in tqdm.tqdm(data_iter, total=num_utts):
                ids = [hparams.tokenizer.piece_to_id(piece) for piece in text]
                encoded = torch.LongTensor(bosl + ids + eosl).to(device)
                encoded = encoded.unsqueeze(0)  # Fake a batch
                tokens_bos = encoded[:,:-1]
                tokens_eos = encoded[:,1:]
                logits = hparams.model(tokens_bos)
                pred = hparams.log_softmax(logits) 
                pred = pred.transpose(1, 2)  # Shape the predictions for the NLL Loss
                loss = torch.nn.functional.nll_loss(pred, tokens_eos, reduction="sum")
                cost_item = loss.cpu().item()
                print(uttid, cost_item, file=fo)
    

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    modules, hparams, device = setup(hparams, run_opts)
    run_test(modules, hparams, device)
