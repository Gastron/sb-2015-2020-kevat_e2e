#!/usr/bin/env python
import pathlib
import re
import matplotlib.pyplot as plt
import cycler
#plt.rcParams.update({"text.usetex": True})

def extract_aed_data(path_to_log):
    matcher = r"epoch: ([^,]*), lr: ([^ ]*) - train loss: ([^ ]*) - valid loss: ([^,]*), valid CER: ([^,]*), valid WER: (.*)"
    matcher = re.compile(matcher)
    data = {"epochs": [],
            "lr": [],
            "train_loss": [],
            "valid_loss": [],
            "valid_cer": [],
            "valid_wer": [],
            }
    with open(path_to_log) as fi:
        for line in fi:
            match = matcher.fullmatch(line.strip())
            if match is None:
                continue
            for key, value in zip(data, match.groups()):
                data[key].append(float(value))
    return data

def extract_hmm_data(path_to_log):
    matcher = r"epoch: ([^,]*), lr: ([^ ]*) - train loss: ([^ ]*) - valid loss: ([^,]*), valid accuracy: (.*)"
    matcher = re.compile(matcher)
    data = {"epochs": [],
            "lr": [],
            "train_loss": [],
            "valid_loss": [],
            "valid_accuracy": [],
            }
    with open(path_to_log) as fi:
        for line in fi:
            match = matcher.fullmatch(line.strip())
            if match is None:
                continue
            for key, value in zip(data, match.groups()):
                data[key].append(float(value))
    return data

def draw_plot(data_aed, data_hmm, savedest):
    sty_cycle = cycler.cycler(c=['tab:blue','tab:orange','tab:green']) + \
            cycler.cycler(ls=["-",":","--"])
    sty_cycle = iter(sty_cycle)
    fig, (ax_aed, ax_hmm) = plt.subplots(1, 2, layout="constrained")
    lns_aed = []
    lns_aed += ax_aed.plot("epochs", "train_loss", label="Training Loss", data=data_aed, **next(sty_cycle))
    lns_aed += ax_aed.plot("epochs", "valid_loss", label="Validation Loss", data=data_aed, **next(sty_cycle))
    ax_aed.set_xlabel("Nominal Epoch")
    ax_aed.set_ylabel("Loss")
    ax_aed.set_ylim([0.,5.])
    ax_aed_twin = ax_aed.twinx()
    lns_aed += ax_aed_twin.plot("epochs", "valid_wer", label="Validation WER", data=data_aed, **next(sty_cycle))
    ax_aed_twin.set_ylabel("WER [%]")
    ax_aed_twin.set_ylim([0.,100.])
    labs_aed = [ln.get_label() for ln in lns_aed]
    ax_aed.legend(lns_aed, labs_aed)
    ax_aed.set_title("AED Training")

    sty_cycle = cycler.cycler(c=['tab:blue','tab:orange','tab:green']) + \
            cycler.cycler(ls=["-",":","--"])
    sty_cycle = iter(sty_cycle)
    lns_hmm = []
    lns_hmm += ax_hmm.plot("epochs", "train_loss", label="Training Loss", data=data_hmm, **next(sty_cycle))
    lns_hmm += ax_hmm.plot("epochs", "valid_loss", label="Validation Loss", data=data_hmm, **next(sty_cycle))
    ax_hmm.set_xlabel("Nominal Epoch")
    ax_hmm.set_ylabel("Loss")
    ax_hmm.set_ylim([0.,1.])
    ax_hmm_twin = ax_hmm.twinx()
    data_hmm["valid_accuracy_p"] = [100.*num for num in data_hmm["valid_accuracy"]]
    lns_hmm += ax_hmm_twin.plot("epochs", "valid_accuracy_p", label="Validation Accuracy [%]", data=data_hmm, **next(sty_cycle))
    ax_hmm_twin.set_ylim([0.,100.])
    ax_hmm_twin.set_ylabel("Accuracy [%]")
    labs_hmm = [ln.get_label() for ln in lns_hmm]
    ax_hmm.legend(lns_hmm, labs_hmm)
    ax_hmm.set_title("HMM / DNN Training")

    fig.savefig(savedest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("aed_log", type=pathlib.Path)
    parser.add_argument("hmm_log", type=pathlib.Path)
    parser.add_argument("fig_out")
    args = parser.parse_args()
    data_aed = extract_aed_data(args.aed_log)
    data_hmm = extract_hmm_data(args.hmm_log)
    draw_plot(data_aed, data_hmm, args.fig_out)
    
