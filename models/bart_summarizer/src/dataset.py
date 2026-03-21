from __future__ import annotations
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from src.config import Config


def load_and_split(cfg: Config) -> DatasetDict:
    df = pd.read_csv(cfg.dataset.csv_path)
    df = df.rename(columns={
        cfg.dataset.article_col: "article",
        cfg.dataset.summary_col: "summary",
    })
    df = df[["article", "summary"]].dropna()

    mask = df["article"].str.contains("\ufffd", regex=False) | \
           df["summary"].str.contains("\ufffd", regex=False)
    df = df[~mask].reset_index(drop=True)
    print(f"Dataset size: {len(df)}")

    test_size = min(cfg.dataset.test_size, int(len(df) * 0.15))
    print(f"Test size: {test_size}")

    dataset = Dataset.from_pandas(df)
    train_test = dataset.train_test_split(
        test_size=test_size,
        seed=cfg.dataset.seed,
    )
    train_val = train_test["train"].train_test_split(
        test_size=cfg.dataset.val_ratio,
        seed=cfg.dataset.seed,
    )

    splits = DatasetDict({
        "train":      train_val["train"],
        "validation": train_val["test"],
        "test":       train_test["test"],
    })
    print(splits)
    return splits


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer,
                     cfg: Config) -> DatasetDict:

    def preprocess(examples):
        model_inputs = tokenizer(
            examples["article"],
            max_length=cfg.model.max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=cfg.model.max_target_length,
            truncation=True,
            padding="max_length",
        )
        labels["input_ids"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print(tokenized)
    return tokenized
