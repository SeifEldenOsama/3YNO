"""## Extract Plan"""

config_yaml_content = """
defaults:
  premise_path: output/premise.json
  output_path: output/plan.json
  logging_level: info
  MODEL:
    engine: mistralai/Mistral-7B-Instruct-v0.3
    tensor_parallel_size: 1
    server_type: vllm
    host: http://localhost
    port: 9741
    prompt_format: openai-chat
    temperature: 0.5
    top_p: 0.9
    frequency_penalty: 0
    presence_penalty: 0
    PLAN:
      SETTING:
        max_tokens: 150
        stop: []
      ENTITY:
        max_attempts: 2
        min_entities: 1
        max_entities: 1
        NAME:
          max_tokens: 16
          stop: ["\n", ",", ":", "("]
        DESCRIPTION:
          max_tokens: 80
      OUTLINE:
        max_attempts: 2
        expansion_policy: breadth-first
        max_depth: 1
        context: ancestors-with-siblings-children
        min_children: 1
        preferred_max_children: 1
        max_children: 1
        EVENT_DEPTH_0:
          max_tokens: 120
        EVENT:
          frequency_penalty: 0.5
          max_tokens: 120
        SCENE:
          context: ancestors-with-siblings
          max_tokens: 80
        ENTITY_DEPTH_0:
          max_tokens: 80
        ENTITY:
          max_tokens: 80
"""


all_confs = recursive_lowercase_keys(yaml.safe_load(config_yaml_content ))
config = Config.load_from_dict(all_confs, ['defaults'])

print("Configuration loaded.")

prompts_json_content = """
{
  "plan": {
    "setting": {
      "instruction": "Create a fun, colorful, and simple setting for a children's story. Use short sentences and words kids understand. Show what it looks like, sounds like, and feels like. Title: {title}//Premise: {premise}",
      "response_prefix": ""
    },
    "entity": {
      "name": {
        "instruction": "Generate only the next character's name for a children's story. Use fun and simple names. Output only the name. Do not repeat previous names. Title: {title}//Premise: {premise}//Setting: {setting}//Existing Characters: {entity_list}",
        "response_prefix": ""
      },
      "description": {
        "instruction": "Describe the character in one simple sentence. Include what they look like and what makes them special. Avoid hard words. Title: {title}//Premise: {premise}//Setting: {setting}//Character: {entity_name}",
        "response_prefix": ""
      }
    },
    "outline": {
      "event_depth_0": {
        "instruction": "Write the first important event of the story in one short, clear sentence for kids. Only describe the event. Title: {title}//Premise: {premise}//Setting: {setting}//Characters: {entities}",
        "response_prefix": ""
      },
      "entity_depth_0": {
        "instruction": "List all main characters appearing in this top-level event. Output them as a comma-separated list. Only use names from the global entity list. Do not invent new names. Title: {title}//Premise: {premise}//Setting: {setting}//Event: {current_event}//Detected Entities: {detected_entities}",
        "response_prefix": ""
      },
      "event": {
        "instruction": "Write the next event in the story as one simple sentence. Keep it fun, clear, and easy to understand for kids. Title: {title}//Premise: {premise}//Setting: {setting}//Characters: {entities}//Outline so far://{context_prefix}",
        "response_prefix": ""
      },
      "scene": {
        "instruction": "Describe where this event happens in one sentence. Keep it easy to picture and kid-friendly. Title: {title}//Premise: {premise}//Setting: {setting}//Characters: {entities}//Event: {current_event}",
        "response_prefix": ""
      },
      "entity": {
        "instruction": "Identify all characters present in this event. Use only names from the main entity list. Return them as a comma-separated list. Title: {title}//Premise: {premise}//Setting: {setting}//Event: {current_event}//Scene: {current_scene}//Detected Entities: {detected_entities}",
        "response_prefix": ""
      }
    }
  }
}

"""
prompts_dict = json.loads(prompts_json_content)
prompts = load_prompts_from_dict(prompts_dict)
print("Prompts loaded and templates created.")

import argparse
import os

from pathlib import Path
import string
from collections.abc import Sequence
from functools import partial
import string
import uuid

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

        def flatten_list(x):
            if isinstance(x, list):
                flat = []
                for item in x:
                    if isinstance(item, list):
                        flat.extend(flatten_list(item))
                    else:
                        flat.append(item)
                return flat
            return [x]

        try:
            setting_items = flatten_list(self.setting)
            setting_str = "\n".join(str(s) for s in setting_items)
        except:
            setting_str = str(self.setting.setting) if self.setting and hasattr(self.setting, 'setting') else str(self.setting)

        try:
            if hasattr(self.entity_list, 'entities'):
                entities_str = str(self.entity_list)
            else:
                entities_str = "\n".join(
                    f"{i+1}. {str(e.name)}: {str(e.description)}"
                    for i, e in enumerate(self.entity_list)
                )
        except Exception as e:
                entities_str = str(self.entity_list)
                logging.error(f"Error formatting entities: {e}")


        try:
            outline_str = str(self.outline)
        except:
            outline_str = str(self.outline)


        return (
            f"{premise_str}\n\n"
            f"Setting:\n{setting_str}\n\n"
            f"Characters and Entities:\n{entities_str}\n\n"
            f"Outline:\n{outline_str}"
        )



    def save(self, path):
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
            name = entity.name
            desc = entity.description

            if isinstance(name, list):
                name = name[0] if name else ""
            if isinstance(desc, list):
                desc = desc[0] if desc else ""

            lines.append(f"{i+1}. {str(name)}: {str(desc)}")

        return "\n\n".join(lines)


    def print_with_full_names(self):
        return '\n\n'.join([f'{i+1}. Full Name: {entity.name}\n\nDescription: {entity.description}' for i, entity in enumerate(self.entities)])

    def __iter__(self):
        return iter(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

    def get_entity_by_name(self, name):
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise ValueError(f'EntityList has no entity named {name}.')


def detect_entities(event, entity_list):
    detected_entities = []

    if isinstance(event, list) and event:
        event = event[0]

    if not isinstance(event, str):
        logging.error(f"Entity detection failed: 'event' is not a string after normalization: {event}")
        return detected_entities

    event_lower = event.lower()

    stopwords_list = stopwords.words('english')

    for entity in entity_list:
        entity_name = entity.name
        if isinstance(entity_name, list) and entity_name:
            entity_name = entity_name[0]

        if not isinstance(entity_name, str):
            continue

        for name_part in entity_name.split():
            name_part_lower = name_part.lower()

            if name_part_lower in stopwords_list:
                continue

            if name_part_lower in event_lower:
                if entity.name not in detected_entities:
                    detected_entities.append(entity.name)
                break
    return detected_entities

def num_to_char(n):
    """Converts a number (1, 2, 3...) to a letter (A, B, C...)."""
    if n < 1:
        return '0'
    return chr(64 + n)
def num_to_roman(n):
     import roman
     return roman.toRoman(n)

class OutlineNode(Sequence):
    def pretty(self):
        """Generates a nicely formatted, hierarchical string representation of the outline."""
        lines = []
        for node in self.depth_first_traverse(include_self=False):
             lines.append(node.format_self())
        return '\n'.join(lines)

    @staticmethod
    def from_dict(d, parent=None):
        node = OutlineNode(d['text'], parent, d['scene'], d['entities'], d['id'])
        node.children = [OutlineNode.from_dict(child, node) for child in d['children']]
        return node

    @staticmethod
    def num_converter(depth):
        if depth == 0:
            return lambda num: ''
        if depth % 3 == 1:
            return str
        elif depth % 3 == 2:
            return num_to_char
        elif depth % 3 == 0:
            return num_to_roman

    @staticmethod
    def indent(depth):
        if depth == 0:
            return ''
        return '\t' * (depth-1)

    def __init__(self, text, parent, scene='', entities=None, id=None):

        if isinstance(text, list):
            if text:
                text = text[0]
            else:
                text = ""

        if not isinstance(text, str):
            text = str(text)

        self.text = text.strip()
        self.entities = entities if entities is not None else []
        self.scene = scene
        self.children = []
        self.parent = parent
        self.id = str(uuid.uuid4()) if id is None else id
        super().__init__()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def to_dict(self):
        return {
            'text': self.text,
            'scene': self.scene,
            'entities': self.entities,
            'children': [child.to_dict() for child in self.children],
            'id': self.id
        }

    def format_self(self):
        if isinstance(self.text, list) and self.text:
            self.text = self.text[0]
        if not isinstance(self.text, str):
            self.text = ""

        scene_text = self.scene
        if isinstance(scene_text, list) and scene_text:
            scene_text = scene_text[0]
        if not isinstance(scene_text, str):
            scene_text = ""

        s = self.number() + self.text

        if len(scene_text) > 0:
            s += ' Scene: ' + scene_text

        return s


    def __str__(self):
        s = f"{self.number()}: {self.text}"

        for child in self.children:
            s += "\n" + str(child)
        return s

    def __len__(self):
        return len(self.children)

    def __getitem__(self, index):
        return self.children[index]

    def get_node_by_id(self, id):
        for node in self.root().depth_first_traverse():
            if node.id == id:
                return node
        return None

    def number(self, depth_shift=0, lookforward=0, convert=True):
        if self.parent is None:
            num = 1
        else:
            try:
                num = self.parent.children.index(self) + 1
            except ValueError:
                num = 1
                for i, child in enumerate(self.parent.children):
                    if child.id == self.id:
                        num = i + 1
                        break

        num += lookforward

        if convert:
            depth = self.depth() + depth_shift
            if depth == 0:
                return ''
            return '\t' * (depth-1) + OutlineNode.num_converter(depth)(num) + '. '

        return num

    def depth(self):
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def root(self):
        if self.parent is None:
            return self
        return self.parent.root()

    def predecessor(self, max_depth=1e8):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        idx = nodes.index(self)
        return nodes[idx-1] if idx > 0 else None

    def successor(self, max_depth=1e8):
        nodes = list(self.root().depth_first_traverse(max_depth=max_depth))
        idx = nodes.index(self)
        return nodes[idx+1] if idx < len(nodes)-1 else None

    def ancestors(self, include_self=False):
        if self.parent is None:
            return [self] if include_self else []
        return self.parent.ancestors(include_self=True) + ([self] if include_self else [])

    def siblings(self, include_self=False):
        if self.parent is None:
            return []
        return [child for child in self.parent.children if (include_self or child != self)]

    def leaves(self):
        if len(self.children) == 0:
            return [self]
        return sum([child.leaves() for child in self.children], [])

    def depth_first_traverse(self, include_self=True, max_depth=1e8):
        if self.depth() <= max_depth and include_self:
            yield self
        for child in self.children:
            yield from child.depth_first_traverse(max_depth=max_depth)

    def breadth_first_traverse(self, include_self=True, max_depth=1e8):
        if self.depth() <= max_depth and include_self:
            yield self
        if self.depth() < max_depth:
            queue = [c for c in self.children]
            while queue:
                n = queue.pop(0)
                yield n
                if n.depth() < max_depth:
                    queue.extend(n.children)

    def context(self, context_type):
        if context_type == 'full':
            selected_nodes = set(list(self.root().depth_first_traverse(include_self=False)))
        elif context_type == 'ancestors':
            selected_nodes = set(self.ancestors(include_self=False))
        elif context_type == 'ancestors-with-siblings':
            ancestors = list(self.ancestors(include_self=True))
            selected_nodes = set(sum([a.siblings(include_self=True) for a in ancestors], []))
        elif context_type == 'ancestors-with-siblings-children':
            ancestors = list(self.ancestors(include_self=True))
            anc_sibs = sum([a.siblings(include_self=True) for a in ancestors], [])
            selected_nodes = set(anc_sibs + sum([node.children for node in anc_sibs], []))
        else:
            raise NotImplementedError()

        prefix = []
        suffix = []
        in_prefix = True

        for node in self.root().depth_first_traverse(include_self=False):
            if node == self:
                in_prefix = False
            elif node in selected_nodes:
                (prefix if in_prefix else suffix).append(node)

        return (
            '\n\n'.join([n.format_self() for n in prefix]),
            '\n\n'.join([n.format_self() for n in suffix])
        )

import time
import string
from functools import partial
import re

def split_numbered_items(text):

    items = re.split(r'\n?\s*\d+\.\s*', text)
    items = [item.strip() for item in items if item.strip()]
    return items


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

    normalized = []

    if plan.entity_list is not None:
        for item in plan.entity_list:
            if isinstance(item, Entity):
                normalized.append(item)

            elif isinstance(item, dict):
                normalized.append(Entity(item["name"], item["description"]))

            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, Entity):
                        normalized.append(sub)
                    elif isinstance(sub, dict):
                        normalized.append(Entity(sub["name"], sub["description"]))

    plan.entity_list = EntityList(normalized)

    def postprocess_name(generated, **kwargs):
        if not isinstance(generated, (list, tuple)) or not generated:
            return [""]

        text = str(generated[0]).strip().split("\n")[0]
        text = re.sub(r'^\d+\.\s*', '', text)
        return [text.strip()]

    def postprocess_entity_description(descriptions, **kwargs):
        desc = descriptions[0].split("\n")[0].strip()
        return [desc]

    name_config = entity_config['name']
    desc_config = entity_config['description']

    while len(plan.entity_list) < entity_config['max_entities']:

        name = llm_client.call_with_retry(
            entity_prompt['name'].format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entity_list=", ".join(e.name for e in plan.entity_list if hasattr(e, 'name') and isinstance(e.name, str)),
            ),
            SamplingConfig.from_config(name_config),
            postprocessor=postprocess_name,
            filter=Filter(lambda s: len(s.strip()) > 1),
            max_attempts=10
        )[0]

        if isinstance(name, list):
           if name:
             name = name[0]
           else:
             name = ""
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
            SamplingConfig.from_config(desc_config),
            postprocessor=postprocess_entity_description,
            filter=Filter(lambda s: len(s.strip()) > 10),
            max_attempts=10
        )[0]

        if isinstance(desc, list):
           if desc:
             desc = desc[0]
           else:
             desc = ""
        desc = str(desc).strip()


        plan.entity_list.entities.append(Entity(name, desc))

    return plan

def generate_outline(plan, llm_client, outline_prompt, outline_config):
    plan.outline = OutlineNode('', None)
    max_nodes = 50
    while True:
        if len(list(plan.outline.depth_first_traverse())) > max_nodes:
            break

        try:
            node_to_expand = select_node_to_expand(plan.outline, outline_config)
        except StopIteration:
            break

        generate_node_subevents(node_to_expand, llm_client, outline_prompt, outline_config, plan, max_attempts=10)

    return plan

def generate_node_subevents(node, llm_client, outline_prompt, outline_config, plan, max_attempts=1):
    context_prefix = ""
    context_suffix = ""
    filter = Filter(lambda x: True)

    def event_postprocessor(events, **kwargs):
        responses = []
        for event in events:
            event = event.strip()
            event = re.sub(r'^\[[^\]]*\]\s*', '', event)
            event = event.split('\n')[0]
            event = event.split('Scene:')[0]
            event = event.split('Characters:')[0]
            event = event.strip()

            if not event:
                event = "Something happens."

            if event[-1] not in ".?!":
                event += "."

            responses.append(event)
        return responses

    if node.depth() == 0:
        event_config = outline_config['event_depth_0']
        event_prompt = outline_prompt['event_depth_0']
    else:
        event_config = outline_config['event']
        event_prompt = outline_prompt['event']

    for _ in range(outline_config['preferred_max_children']):
        new_child = OutlineNode('', node)

        event = llm_client.call_with_retry(
            event_prompt.format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entities=str(plan.entity_list),
                formatted_current_number=new_child.number().rstrip(),
                stripped_current_number=new_child.number().strip(),
                context_prefix=context_prefix,
                context_suffix=context_suffix,
                predecessor_info="",
                successor_info="",
                preferred_max_children=outline_config['preferred_max_children']
            ),
            SamplingConfig.from_config(event_config),
            postprocessor=partial(
                event_postprocessor,
                has_next_indicator="\n" + new_child.number(lookforward=1).strip(),
                current_number=new_child.number().strip()
            ),
            filter=filter,
            max_attempts=max_attempts
        )

        logging.warning(f"Raw LLM event: {event}")

        new_child.text = event[0]
        node.children.append(new_child)

        context_prefix, context_suffix = new_child.context(outline_config['context'])

        if len(node.children) >= outline_config['max_children']:
            break

        filter = Filter(lambda x: True)

        generate_node_scene(
            new_child, llm_client,
            outline_prompt['scene'], outline_config['scene'], plan
        )

        generate_node_entities(
            new_child, llm_client,
            outline_prompt['entity_depth_0'] if node.depth() == 0 else outline_prompt['entity'],
            outline_config['entity_depth_0'] if node.depth() == 0 else outline_config['entity'],
            plan
        )

def generate_node_scene(node, llm_client, scene_prompt, scene_config, plan):
    def scene_postprocessor(scenes, **kwargs):
        clean = []
        for sc in scenes:
            sc = sc.split('\n')[0].split('Characters:')[0].split('Scene:')[-1].strip()
            clean.append(sc)
        return clean

    context_prefix, context_suffix = node.context(scene_config['context'])

    node.scene = llm_client.call_with_retry(
        scene_prompt.format(
            title=plan.premise.title,
            premise=plan.premise.premise,
            setting=plan.setting.setting,
            entities=str(plan.entity_list),
            formatted_current_number=node.number().rstrip(),
            stripped_current_number=node.number().strip(),
            current_event=node.text,
            context_prefix=context_prefix,
            context_suffix=context_suffix
        ),
        SamplingConfig.from_config(scene_config),
        postprocessor=scene_postprocessor,
        filter=Filter(lambda s: len(s.strip()) > 0),
    )[0]

def generate_node_entities(node, llm_client, entity_prompt, entity_config, plan):
    detected = detect_entities(node.text, plan.entity_list)
    if detected:
       node.entities = detected
       return

    def entity_postprocessor(predicted_lists, entity_list, already_detected, **kwargs):
        out = []
        for ents in predicted_lists:
            ents = ents.split('\n')[0].strip().rstrip('.')
            ents = [e.strip() for e in ents.split(',')]
            ents = [e for e in ents if e in [x.name for x in entity_list]]
            ents = list(dict.fromkeys(ents))
            ents = [e for e in ents if e not in already_detected]
            out.append(already_detected + ents)
        return out

    detected = detect_entities(node.text[0], plan.entity_list)
    context_prefix, context_suffix = node.context(entity_config['context'])

    try:
        node.entities = llm_client.call_with_retry(
            entity_prompt.format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entities=str(plan.entity_list),
                formatted_current_number=node.number().rstrip(),
                stripped_current_number=node.number().strip(),
                current_event=node.text,
                current_scene=node.scene,
                context_prefix=context_prefix,
                context_suffix=context_suffix,
                detected_entities=", ".join(detected)
            ),
            SamplingConfig.from_config(entity_config),
            postprocessor=partial(entity_postprocessor,
                                  entity_list=plan.entity_list,
                                  already_detected=detected),
            filter=Filter(lambda l: len(l) > 0),
            max_attempts=20
        )[0]

    except Exception:
        node.entities = detected

def select_node_to_expand(outline, outline_config):
    if outline_config['expansion_policy'] == 'breadth-first':
        for node in outline.breadth_first_traverse(max_depth=outline_config['max_depth'] - 1):
            if len(node.children) == 0:
                return node
        raise StopIteration
    else:
        raise NotImplementedError

try:
    prompts_dict = json.loads(prompts_json_content)
    prompts = load_prompts_from_dict(prompts_dict)
    premise = Premise.load(config['premise_path'])

    plan = Plan(premise)

    plan_config = config['model']['plan']
    plan_prompts = prompts['plan']

    logging.info("Generating setting...")
    plan = generate_setting(
        plan,
        llm_client,
        plan_prompts['setting'],
        plan_config['setting']
    )
    logging.info(f"Generated setting: {plan.setting.setting}")
    torch.cuda.empty_cache()

    logging.info("Generating entities...")
    plan = generate_entities(
        plan,
        llm_client,
        plan_prompts['entity'],
        plan_config['entity']
    )
    logging.info(f"Generated entities: {plan.entity_list}")
    torch.cuda.empty_cache()

    logging.info("Generating outline...")
    plan = generate_outline(
        plan,
        llm_client,
        plan_prompts['outline'],
        plan_config['outline']
    )
    logging.info(f"Generated outline with {len(list(plan.outline.depth_first_traverse()))} nodes.")
    torch.cuda.empty_cache()


    output_path = config['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plan.save(output_path)

    print("\n--- FINAL RESULT ---")
    print(plan)
    print(f"\nPlan object saved to: {output_path}")

except Exception as e:
    logging.error(f"An error occurred during execution. Please check your model configuration and ensure your LLM server is running and accessible. Error: {e}")
    raise e
