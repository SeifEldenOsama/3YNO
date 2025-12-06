!pip install --upgrade pyyaml roman python-Levenshtein transformers scipy langchain langchain-core langchain-community

!pip install -q bitsandbytes accelerate

!pip install -U bitsandbytes

"""## preparing LLM"""

import os
import logging
import json
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
import roman
import Levenshtein
from transformers import AutoTokenizer
from scipy.special import log_softmax
from langchain_core.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
print("Initial imports and logging configured.")

LOCALHOST = 'http://localhost'
DEFAULT_PORT = 8000

class ServerConfig:
    def __init__(self, engine, host, port, server_type, tensor_parallel_size):
        self.engine = engine
        self.host = host
        self.port = port
        self.server_type = server_type
        self.tensor_parallel_size = tensor_parallel_size

    @staticmethod
    def from_config(config):
        return ServerConfig(
            engine=config['engine'],
            host=config['host'],
            port=config.get('port', DEFAULT_PORT),
            server_type=config['server_type'],
            tensor_parallel_size=config['tensor_parallel_size']
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __hash__(self):
        return hash((self.engine, self.host, self.port,
                     self.server_type, self.tensor_parallel_size))

    def __eq__(self, other):
        return (self.engine, self.host, self.port, self.server_type, self.tensor_parallel_size) == (other.engine, other.host, other.port, other.server_type, self.tensor_parallel_size)

def recursive_lowercase_keys(d):
    if type(d) is dict:
        new_d = {}
        for key in d:
            new_d[key.lower()] = recursive_lowercase_keys(d[key])
        return new_d
    else:
        return d

class Config:
    def __init__(self, config, parent=None):
        self.parent_config = parent
        self.config = config
        for key in self.config:
            if type(self.config[key]) is dict:
                self.config[key] = Config(self.config[key], self)

    @staticmethod
    def load_from_dict(all_confs, config_names):
        config = {}
        for name in config_names:
            if ',' in name:
                for n in name.split(','):
                    config.update(all_confs[n])
            else:
                config.update(all_confs[name])

        return Config(config, None)

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        elif self.parent_config is not None:
            return getattr(self.parent_config, name)
        else:
            raise AttributeError(f"Config has no attribute {name}.")

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.config

    def get(self, name, default=None):
        try:
            return self[name]
        except AttributeError:
            return default

tokenizers = {}

def init_logging(logging_level):
    logging_level = logging_level.upper()
    assert logging_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logging.getLogger().setLevel(logging_level)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    try:
        yield
    finally:
        pass

class Filter:
    def __init__(self, filter_func):
        self.filter_func = filter_func

    @staticmethod
    def wrap_preprocessor(preprocessor, filter):
        return Filter(lambda s: filter(preprocessor(s)))

    def __call__(self, *args, **kwargs):
        try:
            return self.filter_func(*args, **kwargs)
        except:
            return self.filter_func(*args)

    def __add__(self, other):
        return Filter(lambda s: self.filter_func(s) and other.filter_func(s))

def min_max_tokens_filter(min_tokens, max_tokens, tokenizer_model_string='gpt2', filter_empty=True):
    global tokenizers
    if tokenizer_model_string not in tokenizers:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_string)
            tokenizers[tokenizer_model_string] = tokenizer
        except Exception as e:
            logging.warning(f"Could not load tokenizer {tokenizer_model_string}. Token filtering will be skipped. Error: {e}")
            return Filter(lambda s: True)
    else:
        tokenizer = tokenizers[tokenizer_model_string]

    filter_func = Filter(lambda s: min_tokens <= len(tokenizer.encode(s.strip())) <= max_tokens)
    if filter_empty:
        filter_func = filter_func + Filter(lambda s: len(s.strip()) > 0)
    return filter_func

def levenshtein_ratio_filter(passages_to_match, threshold=0.8):
    return Filter(lambda s: all([all([Levenshtein.ratio(sub_s, passage) < threshold for passage in passages_to_match]) for sub_s in s.split()]))

def word_filter(word_list):
    return Filter(lambda s: all([word not in s for word in word_list]))

def list_next_number_format_filter():
    bad_regex = re.compile(r'[^]\d+\.')
    return Filter(lambda s: not bad_regex.search(s))

def extract_choice_logprobs(full_completion, choices=['yes', 'no'], default_logprobs=[-1e8, -1e8], case_sensitive=False):
    batch_logprobs = []
    for choice in full_completion['choices']:
        all_logprobs = choice['logprobs']['top_logprobs']
        found = False
        logprobs = [l for l in default_logprobs]
        for token_logprobs in all_logprobs:
            for key, value in token_logprobs.items():
                for i, choice in enumerate(choices):
                    if choice in key or (not case_sensitive and choice.lower() in key.lower()):
                        found = True
                        logprobs[i] = value
            if found:
                break
        batch_logprobs.append(log_softmax(logprobs))
    return batch_logprobs

warned_prompt_format = {'openai_response_prefix': False}

def format_langchain_prompt(langchain_prompt, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in langchain_prompt.input_variables}
    return langchain_prompt.format(**kwargs)

class TemplatePromptBuilder:
    def __init__(self, base_dict):
        self.instruction = PromptTemplate.from_template(template=base_dict['instruction'],)
        self.system_message = PromptTemplate.from_template(template=base_dict['system_message'],) if 'system_message' in base_dict else None
        self.response_prefix = PromptTemplate.from_template(template=base_dict['response_prefix'],) if 'response_prefix' in base_dict else None
        self.output_prefix = PromptTemplate.from_template(template=base_dict['output_prefix'],) if 'output_prefix' in base_dict else None

    def format(self, **kwargs):
        return PromptBuilder(self, **kwargs)

class PromptBuilder:
    def __init__(self, template_prompt_builder, **kwargs):
        self.instruction = format_langchain_prompt(template_prompt_builder.instruction, **kwargs)
        self.system_message = format_langchain_prompt(template_prompt_builder.system_message, **kwargs) \
            if template_prompt_builder.system_message is not None else None
        self.response_prefix = format_langchain_prompt(template_prompt_builder.response_prefix, **kwargs) \
            if template_prompt_builder.response_prefix is not None else None
        self.output_prefix = format_langchain_prompt(template_prompt_builder.output_prefix, **kwargs) \
            if template_prompt_builder.output_prefix is not None else None

    def render_for_llm_format(self, prompt_format):
        if prompt_format not in ['openai-chat', 'llama2-chat', 'none']:
            raise NotImplementedError(f"Prompt format {prompt_format} not implemented.")

        prompt = self.instruction.format().lstrip()

        if prompt_format == 'openai-chat':
            if self.response_prefix is not None:
                global warned_prompt_format
                if warned_prompt_format['openai_response_prefix']:
                    logging.warning(f"Response prefix is not supported for prompt format {prompt_format}. Appending to end of instruction instead.")
                    warned_prompt_format['openai_response_prefix'] = True
                prompt += '\n\n\n\nThe output is already partially generated. Continue from:\n\n' + self.response_prefix.format()
            messages = [{'role': 'user', 'content': prompt}]
            if self.system_message is not None:
                messages = [{'role': 'system', 'content': self.system_message.format()}] + messages
            return messages

        else:
            if prompt_format == 'llama2-chat':
                prompt = '[INST]'
                if self.system_message is not None:
                    prompt += ' <<SYS>>\n' + self.system_message.format() + '\n<</SYS>>\n\n'
                else:
                    prompt += ' '
                prompt += self.instruction.format()
                prompt += '[/INST]' if self.instruction.format()[-1] == ' ' else ' [/INST]'
                if self.response_prefix is not None:
                    prompt += self.response_prefix.format() if self.response_prefix.format()[0] == ' ' else ' ' + self.response_prefix.format()
            else:
                if self.system_message is not None:
                    prompt = self.system_message.format() + '\n\n\n\n' + prompt
                if self.response_prefix is not None:
                    prompt = prompt + '\n\n\n\n' + self.response_prefix.format()
            return prompt

def _create_prompt_templates(prompts):
    for key in prompts:
        assert isinstance(prompts[key], dict)
        if 'instruction' not in prompts[key]:
            _create_prompt_templates(prompts[key])
        else:
            prompts[key] = TemplatePromptBuilder(prompts[key])

def load_prompts_from_dict(prompts_dict):
    prompts = prompts_dict.copy()
    _create_prompt_templates(prompts)
    return prompts

import time
models = {}

class SamplingConfig:
    def __init__(self,
                 server_config,
                 prompt_format,
                 max_tokens=None,
                 temperature=None,
                 top_p=None,
                 frequency_penalty=None,
                 presence_penalty=None,
                 stop=None,
                 n=None,
                 logit_bias=None,
                 logprobs=None):
        self.server_config = server_config
        self.prompt_format = prompt_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.n = n
        self.logit_bias = logit_bias
        self.logprobs = logprobs

    @staticmethod
    def from_config(config):
        return SamplingConfig(
            server_config=ServerConfig.from_config(config),
            prompt_format=config['prompt_format'],
            max_tokens=config.get('max_tokens', None),
            temperature=config.get('temperature', None),
            top_p=config.get('top_p', None),
            frequency_penalty=config.get('frequency_penalty', None),
            presence_penalty=config.get('presence_penalty', None),
            stop=config.get('stop', None),
            n=config.get('n', None),
            logit_bias=config.get('logit_bias', None),
            logprobs=config.get('logprobs', None)
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def dict(self):
        d = {'model': self.server_config.engine}
        for attr in ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'n', 'logit_bias', 'logprobs']:
            if getattr(self, attr) is not None:
                d[attr] = getattr(self, attr)
        return d


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
class LLMClient:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def call_with_retry(self, prompt_builder, sampling_config, postprocessor=None,
                        filter=lambda s: True, max_attempts=5, **kwargs):
        for attempt in range(max_attempts):
            try:
                completions, full_obj = self(prompt_builder, sampling_config, **kwargs)

                if postprocessor:
                    completions = postprocessor(completions)

                completions = [c for c in completions if filter(c)]
                if completions:
                    return completions, full_obj

            except Exception as e:
                print(f"ERROR attempt {attempt+1}: {e}")

        raise RuntimeError("Failed after retries.")

    def __call__(self, prompt_builder, sampling_config, **kwargs):
        messages = prompt_builder.render_for_llm_format(sampling_config.prompt_format)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        max_tokens = sampling_config.max_tokens if sampling_config.max_tokens is not None else 512
        temperature = sampling_config.temperature if sampling_config.temperature is not None else 1.0
        top_p = sampling_config.top_p if sampling_config.top_p is not None else 1.0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0, inputs.input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return [text], None

llm_client = LLMClient(model, tokenizer)
