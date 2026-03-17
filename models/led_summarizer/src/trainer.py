from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
import evaluate
from src.config import Config
from src.dataset import load_and_split, tokenize_dataset


class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step {state.global_step} | loss: {logs['loss']:.4f}")


def build_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        labels      = np.clip(labels,      0, tokenizer.vocab_size - 1)

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v * 100, 4) for k, v in result.items()}

    return compute_metrics


class LEDSummarizerTrainer:
    def __init__(self, cfg: Config, csv_path: str, output_dir: str, volume=None):
        self.cfg        = cfg
        self.csv_path   = csv_path
        self.output_dir = output_dir
        self.volume     = volume

    def _set_seed(self):
        seed = self.cfg.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def run(self):
        self._set_seed()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        t   = self.cfg.training
        m   = self.cfg.model
        cfg = self.cfg
        cfg.dataset.csv_path = self.csv_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        tokenizer = AutoTokenizer.from_pretrained(m.checkpoint)
        model     = LEDForConditionalGeneration.from_pretrained(m.checkpoint).to(device)

        raw_datasets       = load_and_split(cfg)
        tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, cfg)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir                  = self.output_dir,
            num_train_epochs            = t.epochs,
            per_device_train_batch_size = t.per_device_train_batch,
            per_device_eval_batch_size  = t.per_device_eval_batch,
            gradient_accumulation_steps = t.gradient_accum_steps,
            gradient_checkpointing      = t.gradient_checkpointing,
            warmup_steps                = t.warmup_steps,
            weight_decay                = t.weight_decay,
            logging_dir                 = os.path.join(self.output_dir, "logs"),
            logging_strategy            = "steps",
            logging_steps               = t.logging_steps,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            fp16                        = t.fp16,
            report_to                   = "none",
            save_total_limit            = t.save_total_limit,
            predict_with_generate       = True,
            generation_max_length       = t.generation_max_length,
            seed                        = t.seed,
        )

        trainer = Seq2SeqTrainer(
            model            = model,
            args             = training_args,
            train_dataset    = tokenized_datasets["train"],
            eval_dataset     = tokenized_datasets["validation"],
            processing_class = tokenizer,
            data_collator    = data_collator,
            compute_metrics  = build_compute_metrics(tokenizer),
            callbacks        = [LossPrinterCallback()],
        )

        print("Starting training...")
        trainer.train()

        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        print("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        for k, v in test_results.items():
            print(f"{k}: {v}")

        if self.volume:
            self.volume.commit()

        print(f"Training complete. Model saved to: {self.output_dir}")
        return test_results