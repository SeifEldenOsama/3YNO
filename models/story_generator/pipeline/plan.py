import os
import json
import yaml
import logging
import argparse
import string
import uuid
import re
from pathlib import Path
from collections.abc import Sequence
from functools import partial
from utils.config import recursive_lowercase_keys, Config, load_prompts_from_dict, init_logging, SamplingConfig, Filter
from models.llm_client import llm_client
from pipeline.premise import Premise

# Load configurations from external files
CONFIG_DIR = Path(__file__).parent.parent / "configs"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_configs" / "plan_config.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts" / "plan_prompts.json"

with open(MODEL_CONFIG_PATH, 'r') as f:
    all_confs = recursive_lowercase_keys(yaml.safe_load(f))
config = Config.load_from_dict(all_confs, ['defaults'])

with open(PROMPTS_PATH, 'r') as f:
    prompts_dict = json.load(f)
prompts = load_prompts_from_dict(prompts_dict)

print("Configuration and prompts loaded from external files.")

class Setting:
    def __init__(self, setting):
        if isinstance(setting, list) and setting:
            self.setting = setting[0]
        else:
            self.setting = setting

    def __str__(self):
        if isinstance(self.setting, str):
            return self.setting
        return str(self.setting)

class Plan:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        premise = Premise(data['premise']['title'], data['premise']['premise'])
        setting = Setting(data['setting'])

        flat = []
        def add_item(obj):
            if obj is None:
                return
            if isinstance(obj, Entity):
                flat.append(obj)
            elif isinstance(obj, dict):
                if "name" in obj and "description" in obj:
                    flat.append(Entity(obj["name"], obj["description"]))
            elif isinstance(obj, list):
                for sub in obj:
                    add_item(sub)

        add_item(data["entities"])
        entity_list = EntityList(flat)
        outline = OutlineNode.from_dict(data['outline'])
        return Plan(premise, setting, entity_list, outline)

    def __init__(self, premise, setting=None, entity_list=None, outline=None):
        self.premise = premise
        self.setting = setting
        self.entity_list = entity_list
        self.outline = outline

    def __str__(self):
        premise_str = str(self.premise) if self.premise is not None else ""
        setting_str = str(self.setting)
        entities_str = str(self.entity_list)
        outline_str = str(self.outline)

        return (
            f"{premise_str}\n\n"
            f"Setting:\n{setting_str}\n\n"
            f"Characters and Entities:\n{entities_str}\n\n"
            f"Outline:\n{outline_str}"
        )

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'premise': {
                    'title': self.premise.title,
                    'premise': self.premise.premise
                },
                'setting': self.setting.setting,
                'entities': [{
                    'name': entity.name,
                    'description': entity.description
                } for entity in self.entity_list],
                'outline': self.outline.to_dict()
            }, f, indent=4)

try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords

class Entity:
    def __init__(self, name, description):
        if isinstance(name, list):
            name = name[0] if name else ""
        self.name = str(name).strip()
        if isinstance(description, list):
            description = description[0] if description else ""
        self.description = str(description).strip()

class EntityList:
    def __init__(self, entities=None):
        self.entities = entities if entities is not None else []

    def __len__(self):
        return len(self.entities)

    def __str__(self):
        lines = []
        for i, entity in enumerate(self.entities):
            lines.append(f"{i+1}. {str(entity.name)}: {str(entity.description)}")
        return "\n\n".join(lines)

    def __iter__(self):
        return iter(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

def detect_entities(event, entity_list):
    detected_entities = []
    if isinstance(event, list) and event:
        event = event[0]
    if not isinstance(event, str):
        return detected_entities
    event_lower = event.lower()
    for entity in entity_list:
        if entity.name.lower() in event_lower:
            detected_entities.append(entity.name)
    return list(dict.fromkeys(detected_entities))

class OutlineNode:
    @staticmethod
    def from_dict(data, parent=None):
        node = OutlineNode(data['text'], parent)
        node.scene = data.get('scene', '')
        node.entities = data.get('entities', [])
        for child_data in data.get('children', []):
            node.children.append(OutlineNode.from_dict(child_data, node))
        return node

    def __init__(self, text, parent=None):
        self.text = text
        self.parent = parent
        self.children = []
        self.scene = ''
        self.entities = []

    def depth(self):
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def root(self):
        if self.parent is None:
            return self
        return self.parent.root()

    def number(self, lookforward=0):
        if self.parent is None:
            return ""
        num = self.parent.children.index(self) + 1 + lookforward
        return f"{self.parent.number()}{num}."

    def to_dict(self):
        return {
            'text': self.text,
            'scene': self.scene,
            'entities': self.entities,
            'children': [c.to_dict() for c in self.children]
        }

    def format_self(self):
        return f"{self.number()} {self.text}"

    def __str__(self):
        res = self.format_self()
        for child in self.children:
            res += "\n" + str(child)
        return res

    def depth_first_traverse(self, include_self=True):
        if include_self:
            yield self
        for child in self.children:
            yield from child.depth_first_traverse()

    def ancestors(self, include_self=True):
        if include_self:
            yield self
        if self.parent is not None:
            yield from self.parent.ancestors()

    def siblings(self, include_self=True):
        if self.parent is None:
            return [self] if include_self else []
        return [c for c in self.parent.children if include_self or c != self]

    def context(self, context_type):
        ancestors = list(self.ancestors(include_self=True))
        anc_sibs = sum([a.siblings(include_self=True) for a in ancestors], [])
        selected_nodes = set(anc_sibs + sum([node.children for node in anc_sibs], []))
        
        prefix = []
        in_prefix = True
        for node in self.root().depth_first_traverse(include_self=False):
            if node == self:
                in_prefix = False
            elif node in selected_nodes:
                if in_prefix:
                    prefix.append(node)
        return '\n\n'.join([n.format_self() for n in prefix]), ""

def select_node_to_expand(root, config):
    for node in root.depth_first_traverse():
        if node.depth() < config['max_depth'] and len(node.children) < config['max_children']:
            return node
    raise StopIteration()

def generate_setting(plan, llm_client, setting_prompt, setting_config):
    plan.setting = Setting(
        llm_client.call_with_retry(
            setting_prompt.format(
                title=plan.premise.title,
                premise=plan.premise.premise
            ),
            SamplingConfig.from_config(setting_config),
            filter=Filter(lambda s: len(s.strip()) > 50),
            max_attempts=10
        )[0]
    )
    return plan

def generate_entities(plan, llm_client, entity_prompt, entity_config):
    plan.entity_list = EntityList([])
    while len(plan.entity_list) < entity_config['max_entities']:
        name = llm_client.call_with_retry(
            entity_prompt['name'].format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entity_list=", ".join(e.name for e in plan.entity_list),
            ),
            SamplingConfig.from_config(entity_config['name']),
            max_attempts=10
        )[0]
        name = str(name).strip()
        if name in [e.name for e in plan.entity_list]:
            break
        desc = llm_client.call_with_retry(
            entity_prompt['description'].format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entity_name=name
            ),
            SamplingConfig.from_config(entity_config['description']),
            max_attempts=10
        )[0]
        desc = str(desc).strip()
        plan.entity_list.entities.append(Entity(name, desc))
    return plan

def generate_outline(plan, llm_client, outline_prompt, outline_config):
    plan.outline = OutlineNode('', None)
    max_nodes = 5
    while len(list(plan.outline.depth_first_traverse())) < max_nodes:
        try:
            node_to_expand = select_node_to_expand(plan.outline, outline_config)
        except StopIteration:
            break
        generate_node_subevents(node_to_expand, llm_client, outline_prompt, outline_config, plan)
    return plan

def generate_node_subevents(node, llm_client, outline_prompt, outline_config, plan):
    if node.depth() == 0:
        event_config = outline_config['event_depth_0']
        event_prompt = outline_prompt['event_depth_0']
    else:
        event_config = outline_config['event']
        event_prompt = outline_prompt['event']

    context_prefix, _ = node.context(outline_config['context'])
    new_child = OutlineNode('', node)
    event = llm_client.call_with_retry(
        event_prompt.format(
            title=plan.premise.title,
            premise=plan.premise.premise,
            setting=plan.setting.setting,
            entities=str(plan.entity_list),
            context_prefix=context_prefix,
            current_event="",
            detected_entities=""
        ),
        SamplingConfig.from_config(event_config),
        max_attempts=5
    )[0]
    new_child.text = event.strip()
    node.children.append(new_child)
    
    generate_node_scene(new_child, llm_client, outline_prompt['node_scene'], outline_config['scene'], plan)
    generate_node_entities(new_child, llm_client, 
                           outline_prompt['entity_depth_0'] if node.depth() == 0 else outline_prompt['node_entity'],
                           outline_config['entity_depth_0'] if node.depth() == 0 else outline_config['entity'], 
                           plan)

def generate_node_scene(node, llm_client, scene_prompt, scene_config, plan):
    context_prefix, _ = node.context(scene_config['context'])
    node.scene = llm_client.call_with_retry(
        scene_prompt.format(
            title=plan.premise.title,
            premise=plan.premise.premise,
            setting=plan.setting.setting,
            entities=str(plan.entity_list),
            current_event=node.text,
            context_prefix=context_prefix
        ),
        SamplingConfig.from_config(scene_config),
    )[0].strip()

def generate_node_entities(node, llm_client, entity_prompt, entity_config, plan):
    detected = detect_entities(node.text, plan.entity_list)
    if detected:
        node.entities = detected
        return
    node.entities = []

if __name__ == "__main__":
    try:
        init_logging(config.logging_level)
        logging.info("Loading premise...")
        premise_path = config['premise_path']
        premise = Premise.load(premise_path)
        plan = Plan(premise)
        
        logging.info("Generating setting...")
        generate_setting(plan, llm_client, prompts['plan']['setting'], config['model']['plan']['setting'])
        
        logging.info("Generating entities...")
        generate_entities(plan, llm_client, prompts['plan']['entity'], config['model']['plan']['entity'])
        
        logging.info("Generating outline...")
        generate_outline(plan, llm_client, prompts['plan']['outline'], config['model']['plan']['outline'])
        
        output_path = config['output_path']
        plan.save(output_path)
        print(f"Plan saved to {output_path}")
    except Exception as e:
        logging.error(f"Plan generation failed: {e}")
        raise e
