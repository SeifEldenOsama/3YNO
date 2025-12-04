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
        with open(path, 'w') as f:
            json.dump({
                'title': self.title,
                'premise': self.premise
            }, f, indent=4)

config_yaml_content = """
# MODEL ARGS:
# for model server / sampling args, you can put them under MODEL to be shared,
# or under the specific pipeline steps (TITLE / PREMISE) for specific options. unspecified sampling args will inherit from the ancestor.
# the format is the same as the openai API, since VLLM also provides and openai-style API.

defaults:
  output_path: output/premise.json
  logging_level: info # debug, info, warning, error, critical
  MODEL:
    engine: mistralai/Mistral-7B-Instruct-v0.3 # <-- SET TO AN OPENAI COMPLETION MODEL
    tensor_parallel_size: 1
    server_type: vllm # <-- THIS MUST BE 'openai'
    host: http://localhost # These lines are now ignored, but kept for completeness
    port: 9741
    prompt_format: openai-chat # <-- USE 'none' for completion models like gpt-3.5-turbo-instruct
    temperature: 1.2
    top_p: 0.99
    frequency_penalty: 0
    presence_penalty: 0
    TITLE:
      max_tokens: 64
      stop: []
    PREMISE:
      max_tokens: 512
      stop: ["\\n"]
# ---------------------------------------------

# ---------------------------------------------
"""


all_confs = recursive_lowercase_keys(yaml.safe_load(config_yaml_content ))
config = Config.load_from_dict(all_confs, ['defaults'])

print("Configuration loaded. Please ensure you have set your OPENAI_API_KEY in a previous cell.")

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

educational_summary_input = """
Plants are living things that need care to grow strong and healthy. Every plant starts as a tiny seed. When the seed is placed in soil and given water, it begins to wake up. Soon, small roots grow down into the soil to drink water and collect minerals. After that, a little stem grows upward, reaching for the sunlight.

Plants use sunlight to make their own food in a process called photosynthesis. This helps them grow leaves, flowers, and sometimes fruits or vegetables. Different plants need different amounts of water and sunlight, but all of them need love, attention, and patience. By taking care of plants, children learn responsibility and understand how nature works around them.
"""

prompts_json_content = """
{
    "title": {
        "instruction": "Write a fun and engaging title for a children's short story from this summary: {educational_summary_input}.",
        "response_prefix": ""
    },
    "premise": {
        "instruction": "You MUST NOT write a title. Only write a story premise. Create one paragraph describing the world, the character, and the main problem or adventure. Do NOT include the word 'Title' or any heading. Educational summary: {educational_summary_input}.",
        "response_prefix": ""
    }
}

"""
prompts_dict = json.loads(prompts_json_content)
prompts = load_prompts_from_dict(prompts_dict)
print("Prompts loaded and templates created.")

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    premise.save(output_path)

    print("\n--- FINAL RESULT ---")
    print(premise)
    print(f"\nPremise object saved to: {output_path}")

except Exception as e:
    logging.error(f"An error occurred during execution. Please check your model configuration and ensure your LLM server is running and accessible. Error: {e}")
    raise e