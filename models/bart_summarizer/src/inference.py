from __future__ import annotations
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from src.config import Config


class BARTSummarizerInference:
    def __init__(self, cfg: Config, model_path: str | None = None):
        self.cfg        = cfg
        self.model_path = model_path or cfg.output.local_dir
        self.model      = None
        self.tokenizer  = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model     = BartForConditionalGeneration.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        if self.model is None:
            self.load()

        m = self.cfg.model
        i = self.cfg.inference

        inputs = self.tokenizer(
            text,
            max_length=m.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                num_beams      = i.num_beams,
                max_length     = i.max_length,
                early_stopping = True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_batch(self, texts: list[str]) -> list[str]:
        return [self.summarize(t) for t in texts]

    def summarize_csv(self, csv_path: str, output_path: str | None = None) -> str:
        import pandas as pd
        df = pd.read_csv(csv_path)
        col = self.cfg.dataset.article_col
        df["generated_summary"] = df[col].apply(self.summarize)

        out = output_path or os.path.join(self.cfg.inference.output_dir, "summaries.csv")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved summaries to: {out}")
        return out
