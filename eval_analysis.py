#!/usr/bin/env/python3
"""Finnish Parliament ASR
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds
from glob import glob
import io
import torchaudio
sys.path.append("local/")
from make_shards import segments_to_output, wavscp_to_output, text_to_output, sync_streams
import pathlib

# Brain class for speech recognition training
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats = self.hparams.compute_features(batch.wav.data)
        feats = self.modules.normalize(feats, batch.wav.lengths)

        # Running the encoder (prevent propagation to feature extraction)
        encoded_signal = self.modules.encoder(feats.detach())
        
        predictions, _ = self.hparams.test_search(
            encoded_signal, batch.wav.lengths
        )

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
        predictions = [
                [token for token in pred if token not in specials]
                for pred in predictions
        ]
        predicted_words = [
            self.hparams.tokenizer.decode_ids(prediction).split(" ")
            for prediction in predictions
        ]
        target_words = [words.split(" ") for words in batch.trn]

        # Monitor word error rate and character error rated at
        # valid and test time.
        self.wer_metric.append(batch.__key__, predicted_words, target_words)
        self.cer_metric.append(batch.__key__, predicted_words, target_words)

        return torch.tensor([0.])

    def on_stage_start(self, stage, epoch):
        self.cer_metric = self.hparams.cer_computer()
        self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {}
        stage_stats["CER"] = self.cer_metric.summarize("error_rate")
        stage_stats["WER"] = self.wer_metric.summarize("error_rate")
        with open(self.hparams.wer_file, "w") as w:
            self.wer_metric.write_stats(w)
        with open(self.hparams.decode_text_file, "w") as fo:
            for utt_details in self.wer_metric.scores:
                print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)


class KaldiData(torch.utils.data.IterableDataset):
    def __init__(self, datadir):
        self.datadir = pathlib.Path(datadir)
        self.iterator = None
        if (self.datadir / "segments").exists():
            self.length = self._count_scp_lines(self.datadir / "segments")
            self.dirtype = "segments"
        else:
            self.length = self._count_scp_lines(self.datadir / "wav.scp")
            self.dirtype = "wavscp"

    def __iter__(self):
        if self.dirtype == "segments":
            iterators = [segments_to_output(self.datadir / "segments", self.datadir / "wav.scp")]
        else:
            iterators = [wavscp_to_output(self.datadir / "wav.scp")]
        iterators.append(text_to_output(self.datadir / "text"))
        self.iterator = sync_streams(iterators, maxskip=0)
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        data_point = {}
        for uttid, output in next(self.iterator):
            if "__key__" not in data_point:
                data_point["__key__"] = uttid
            for key, data in output.items():
                if isinstance(data, dict):
                    to_update = data_point.setdefault(key, {})
                    to_update.update(data)
                else:
                    data_point[key] = data
        return {"__key__": data_point["__key__"], 
                "wav": data_point["audio.pth"], 
                "trn": data_point["transcript.txt"]}
        
    @staticmethod
    def _count_scp_lines(scpfile):
        lines = 0
        with open(scpfile) as fin:
            for _ in fin:
                lines += 1
        return lines

if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    hparams["test_data_id"] = "analysis"

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    analysisdata = KaldiData(hparams["analysis_datadir"])
    analysisdata = torch.utils.data.DataLoader(analysisdata, 
            batch_size=12, 
            collate_fn=sb.dataio.batch.PaddedBatch)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = asr_brain.evaluate(
        test_set=analysisdata,
        min_key="WER",
    )
