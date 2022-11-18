#!/usr/bin/env python3

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

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class ShallowFusion(sb.Brain):
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

    def on_evaluate_start(self, max_key=None, min_key=None):
        lm_ckpt = self.hparams.lm_ckpt_finder.find_checkpoint(min_key="loss")
        self.hparams.lm_pretrainer.collect_files(lm_ckpt.path)
        self.hparams.lm_pretrainer.load_collected(self.device)
        asr_ckpt = self.hparams.asr_ckpt_finder.find_checkpoint(min_key="WER")
        self.hparams.asr_pretrainer.collect_files(asr_ckpt.path)
        self.hparams.asr_pretrainer.load_collected(self.device)

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """

    def tokenize(sample):
        text = sample["trn"]
        # quick hack for one sample in text of test2021:
        text = text.replace(" <NOISE>", "")
        fulltokens = torch.LongTensor(
                [hparams["bos_index"]] + hparams["tokenizer"].encode(text) + [hparams["eos_index"]]
        )
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample
    
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )
    testseen = (
            wds.WebDataset(hparams["test_seen_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )
    testunseen = (
            wds.WebDataset(hparams["test_unseen_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )
    test2021 = (
            wds.WebDataset(hparams["test_2021_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )

    test_speecon = (
            wds.WebDataset(hparams["test_speecon_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth", meta="meta.json")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )

    test_yle = (
            wds.WebDataset(hparams["test_yle_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth", meta="meta.json")
            .map(tokenize)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )

    normalizer = sb.dataio.preprocess.AudioNormalizer()
    def normalize_audio(sample):
        signal = sample["wav"]
        samplerate = sample["meta"]["samplerate"]
        sample["wav"] = normalizer(signal, samplerate)
        sample["meta"]["samplerate"] = normalizer.sample_rate 
        return sample

    test_lp= (
            wds.WebDataset(hparams["test_lp_shards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth", meta="meta.json")
            .map(tokenize)
            .map(normalize_audio)
            .batched(
                batchsize=hparams["validbatchsize"], 
                collation_fn=sb.dataio.batch.PaddedBatch,
                partial=True
            )
    )
    
    analysis_uttids = []
    with open(hparams["analysis_datadir"] + "/utt2spk") as fin:
        for line in fin:
            uttid, _ = line.strip().split()
            # HACK: WebDataset cannot handle periods in uttids:
            uttid = uttid.replace(".", "")
            analysis_uttids.append(uttid)
    analysis_uttids = set(analysis_uttids)
    def analysis_select(sample):
        return sample["__key__"] in analysis_uttids

         
    analysisdata = (
            wds.WebDataset(hparams["fullshards"][hparams["aindex"]::hparams["asplits"]])
            .decode()
            .select(analysis_select)
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                sampler_kwargs={"target_batch_numel": 640000,"max_batch_numel": 1000000},
                len_key='wav'
            )
    )

    
    return {"valid": validdata, "test-seen": testseen,
            "test-unseen": testunseen, "test2021": test2021,
            "test-speecon": test_speecon, "test-yle": test_yle,
            "test-lp": test_lp, "analysis": analysisdata}
    

if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ShallowFusion(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets[hparams["test_data_id"]],
    )
