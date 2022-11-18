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

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class TrafoASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.wav
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # augmentation:
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )
        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(ctc_logits)
        else:
            p_ctc = None

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)
        #_, max_indices = torch.sort(p_seq, dim=2, descending=True)
        #for timestep, indices in enumerate(max_indices[0]):
        #    print("Time:", timestep)
        #    for i, ind in enumerate(indices[:2]):
        #        print("\tTop", i, self.hparams.tokenizer.id_to_piece(ind.item()), p_seq[0,timestep,ind].exp())
        #import sys; sys.exit()
        
        if stage == sb.Stage.TRAIN:
            hyps = None 
        elif stage == sb.Stage.VALID:
            hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        return p_ctc, p_seq, wav_lens, hyps 

    def is_ctc_active(self, stage):
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.__key__
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
    
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        if self.is_ctc_active(stage):
            loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
            loss = (
                self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * loss_seq
            )
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index, self.hparams.pad_index]
            # Decode token terms to words
            # NOTE -1 here for padding!
            hyps = [
                    [token -1 for token in pred if token not in specials]
                    for pred in hyps 
            ]
            predicted_words = [
                self.hparams.tokenizer.decode_ids(utt_seq).split() for utt_seq in hyps
            ]
            target_words = [sentence.split() for sentence in batch.trn]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # anneal lr every update, first
            self.hparams.noam_annealing(self.optimizer)

            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()


        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.acc_metric = self.hparams.acc_computer()

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
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                max_keys=["ACC"],
                num_to_keep=self.hparams.ckpts_to_keep,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            if hasattr(self.hparams, "decode_text_file"):
                with open(self.hparams.decode_text_file, "w") as fo:
                    for utt_details in self.wer_metric.scores:
                        print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)


    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            #self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")



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
        fulltokens = torch.LongTensor(
                [hparams["bos_index"]] + hparams["tokenizer"].encode(text) + [hparams["eos_index"]]
        )
        # Pad all inputs by one!
        fulltokens += 1
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample
    
    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    if "valid_dynamic_batch_kwargs" in hparams:
        validdata = (
                wds.WebDataset(hparams["validshards"])
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .then(
                    sb.dataio.iterators.dynamic_bucketed_batch,
                    drop_end=False,
                    **hparams["valid_dynamic_batch_kwargs"]
                )
        )
    else:
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
    validdataall = (
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

    return {"train": traindata, "valid": validdata, 
            "validall":validdataall, "test-seen": testseen, "test-unseen": testunseen}



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

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Pretrain if defined:
    if "pretrainer" in hparams:
        ckpt = hparams["ckpt_finder"].find_checkpoint(min_key="WER")
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = TrafoASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs = hparams["train_loader_kwargs"],
        valid_loader_kwargs = hparams["valid_loader_kwargs"]
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets[hparams["test_data_id"]],
        max_key=hparams["test_max_key"],
    )
