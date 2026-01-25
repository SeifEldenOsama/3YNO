import os
import json
import yaml
import logging
from pathlib import Path
from utils.config import recursive_lowercase_keys, Config, load_prompts_from_dict, init_logging, SamplingConfig, min_max_tokens_filter
from models.llm_client import llm_client

class Premise:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return Premise(data['title'], data['premise'])

    def __init__(self, title=None, premise=None):
        self.title = title
        self.premise = premise

    def __str__(self):
        return f'Title: {self.title}\n\nPremise: {self.premise}'

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'title': self.title,
                'premise': self.premise
            }, f, indent=4)

# Load configurations from external files
CONFIG_DIR = Path(__file__).parent.parent / "configs"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_configs" / "premise_config.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts" / "premise_prompts.json"
DATA_PATH = CONFIG_DIR / "data" / "educational_summary.txt"

with open(MODEL_CONFIG_PATH, 'r') as f:
    all_confs = recursive_lowercase_keys(yaml.safe_load(f))
config = Config.load_from_dict(all_confs, ['defaults'])

with open(PROMPTS_PATH, 'r') as f:
    prompts_dict = json.load(f)
prompts = load_prompts_from_dict(prompts_dict)

with open(DATA_PATH, 'r') as f:
    educational_summary_input = f.read()

print("Configuration, prompts, and data loaded from external files.")

def generate_title(premise_object, title_prompts, title_config, llm_client):
    title = llm_client.call_with_retry(
        title_prompts.format(educational_summary_input=educational_summary_input),
        SamplingConfig.from_config(title_config),
        filter=min_max_tokens_filter(0, title_config['max_tokens'])
    )[0]
    premise_object.title = title
    return premise_object

def generate_premise(premise_object, premise_prompts, premise_config, llm_client):
    premise = llm_client.call_with_retry(
        premise_prompts.format(
            title=premise_object.title,
            educational_summary_input=educational_summary_input
        ),
        SamplingConfig.from_config(premise_config),
        filter=min_max_tokens_filter(0, premise_config['max_tokens'])
    )[0]
    premise_object.premise = premise
    return premise_object

if __name__ == "__main__":
    try:
        init_logging(config.logging_level)
        logging.info("Starting premise generation...")

        premise = Premise()

        logging.info("Generating title...")
        generate_title(premise, prompts['title'], config['model']['title'], llm_client)
        logging.info(f'Generated title: {premise.title}')

        logging.info("Generating premise...")
        generate_premise(premise, prompts['premise'], config['model']['premise'], llm_client)
        logging.info(f'Generated premise: {premise.premise}')

        output_path = config['output_path']
        premise.save(output_path)

        print("\n--- FINAL RESULT ---")
        print(premise)
        print(f"\nPremise object saved to: {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise e
