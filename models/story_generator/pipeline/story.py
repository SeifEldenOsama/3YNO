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

prompts_story_json_content = """
{
  "story": {
    "write": {
        "instruction": "Write the next passage of the story, focusing on the outline event, and maintaining the tone and setting.\n\nStory so far:\n{story_so_far}\n\nOutline Event:\n{outline_event}\n\nScene:\n{scene}\n\nCharacters:\n{entities}\n\nWrite a coherent passage:",
        "response_prefix": ""
    },
    "score": {
        "instruction": "Rate the quality of the following passage on clarity, consistency, emotion, pacing, and creativity. Return a single number from 1 to 10.\n\nPassage:\n{passage}\n\nScore:",
        "response_prefix": ""
    },
    "summarize": {
        "instruction": "Summarize the following passage into 2–3 sentences.\n\nPassage:\n{passage}\n\nSummary:",
        "response_prefix": ""
    }
  }
}
"""

story_prompts_dict = json.loads(prompts_story_json_content)
story_prompts = load_prompts_from_dict(story_prompts_dict)
print("Story Prompts loaded and templates created.")

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

story_config_yaml = """
model:
  engine: "mistralai/Mistral-7B-Instruct-v0.3"
  host: "http://localhost"
  port: 8000
  server_type: vllm
  tensor_parallel_size: 1

  story:
    write:
      max_tokens: 350
      temperature: 0.7
      top_p: 0.9
      prompt_format: openai-chat
    score:
      max_tokens: 20
      temperature: 0.1
      top_p: 0.5
      prompt_format: openai-chat
    summarize:
      max_tokens: 80
      temperature: 0.3
      top_p: 0.9
      prompt_format: openai-chat

output_path: "outputs/story.json"

"""

story_config = Config(yaml.safe_load(story_config_yaml), None)
print("Story configuration loaded.")

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

        logging.info("Configuration and prompts are already loaded globally.")

        logging.info("Generating final story...")
        story = generate_story(
            plan,
            llm_client,
            story_prompts,
            story_config["model"]["story"]
        )

        output_path = story_config["output_path"]
        story.save(output_path)

        print("\n--- FINAL STORY GENERATED SUCCESSFULLY ---")
        print(f"Saved to {output_path}")
        print("\n--- FINAL STORY ---")
        print(story)

    except Exception as e:
        logging.error(f"Story Generation failed: {e}")
        raise e