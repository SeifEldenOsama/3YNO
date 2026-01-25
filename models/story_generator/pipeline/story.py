import os
import json
import yaml
import logging
from pathlib import Path
from utils.config import recursive_lowercase_keys, Config, load_prompts_from_dict, init_logging, SamplingConfig
from models.llm_client import llm_client
from pipeline.plan import Plan

class Story:
    def __init__(self):
        self.passages = []

    def add_passage(self, passage_dict):
        self.passages.append(passage_dict)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "story": self.passages
            }, f, indent=2, ensure_ascii=False)

    def __str__(self):
        all_text = "\n\n".join([p["text"] for p in self.passages])
        return all_text

# Load configurations from external files
CONFIG_DIR = Path(__file__).parent.parent / "configs"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_configs" / "story_config.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts" / "story_prompts.json"

with open(MODEL_CONFIG_PATH, 'r') as f:
    story_config_data = yaml.safe_load(f)
story_config = Config(story_config_data, None)

with open(PROMPTS_PATH, 'r') as f:
    story_prompts_dict = json.load(f)
story_prompts = load_prompts_from_dict(story_prompts_dict)

print("Story configuration and prompts loaded from external files.")

class StoryWriter:
    def __init__(self, llm_client, prompts, config):
        self.llm = llm_client
        self.prompts = prompts
        self.config_write = SamplingConfig.from_config(config["write"])
        self.config_score = SamplingConfig.from_config(config["score"])
        self.config_summarize = SamplingConfig.from_config(config["summarize"])

    def generate_passage(self, story_so_far, outline_event, scene, entities):
        prompt_builder = self.prompts["write"].format(
            story_so_far=story_so_far,
            outline_event=outline_event,
            scene=scene,
            entities=", ".join(entities)
        )
        result = self.llm.call_with_retry(
            prompt_builder,
            self.config_write,
            max_attempts=3
        )[0]
        if isinstance(result, list):
            result = result[0]
        return result

    def score_passage(self, passage):
        prompt_builder = self.prompts["score"].format(
            passage=passage
        )
        result = self.llm.call_with_retry(
            prompt_builder,
            self.config_score,
            max_attempts=2
        )[0]
        if isinstance(result, list):
            result = result[0]
        return result.strip()

    def summarize_passage(self, passage):
        prompt_builder = self.prompts["summarize"].format(
            passage=passage
        )
        result = self.llm.call_with_retry(
            prompt_builder,
            self.config_summarize,
            max_attempts=2
        )[0]
        if isinstance(result, list):
            result = result[0]
        return result

def generate_story(plan, llm_client, prompts, config):
    story_writer = StoryWriter(
        llm_client=llm_client,
        prompts=prompts["story"],
        config=config["model"]["story"]
    )
    story = Story()
    logging.info("Beginning story generation from outline...")
    outline_nodes = list(plan.outline.depth_first_traverse())

    story_text_so_far = ""
    for node in outline_nodes:
        if not node.text: continue
        logging.info(f"Generating passage for node {node.number()}: {node.text}")
        passage = story_writer.generate_passage(
            story_so_far=story_text_so_far,
            outline_event=node.text,
            scene=node.scene,
            entities=node.entities
        )
        summary = story_writer.summarize_passage(passage)
        score = story_writer.score_passage(passage)

        story.add_passage({
            "event_number": node.number(),
            "text": passage,
            "summary": summary,
            "score": score,
            "entities": node.entities,
            "scene": node.scene
        })
        story_text_so_far += "\n" + passage
    return story

if __name__ == "__main__":
    try:
        logging.info("Loading plan...")
        plan_path = "output/plan.json"
        plan = Plan.load(plan_path)
        
        logging.info("Generating final story...")
        story = generate_story(
            plan,
            llm_client,
            story_prompts,
            story_config["model"]["story"]
        )

        output_path = story_config["output_path"]
        story.save(output_path)
        print(f"Final story saved to {output_path}")
    except Exception as e:
        logging.error(f"Story Generation failed: {e}")
        raise e
