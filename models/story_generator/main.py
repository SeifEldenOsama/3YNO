import os
import logging
import json
from pipeline.premise import Premise, generate_title, generate_premise, prompts as premise_prompts, config as premise_config
from pipeline.plan import Plan, generate_setting, generate_entities, generate_outline, prompts as plan_prompts, config as plan_config
from pipeline.story import generate_story, story_prompts, story_config
from models.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """
    Runs the full 3YNO story generation pipeline using external configurations.
    """
    try:
        # 1. Premise Generation
        logging.info("Step 1: Generating Premise...")
        premise = Premise()
        generate_title(premise, premise_prompts['title'], premise_config['model']['title'], llm_client)
        generate_premise(premise, premise_prompts['premise'], premise_config['model']['premise'], llm_client)
        
        premise_path = premise_config['output_path']
        premise.save(premise_path)
        logging.info(f"Premise saved to {premise_path}")

        # 2. Plan Generation
        logging.info("Step 2: Generating Story Plan (Setting, Characters, Outline)...")
        plan = Plan(premise)
        generate_setting(plan, llm_client, plan_prompts['plan']['setting'], plan_config['model']['plan']['setting'])
        generate_entities(plan, llm_client, plan_prompts['plan']['entity'], plan_config['model']['plan']['entity'])
        generate_outline(plan, llm_client, plan_prompts['plan']['outline'], plan_config['model']['plan']['outline'])
        
        plan_path = plan_config['output_path']
        plan.save(plan_path)
        logging.info(f"Plan saved to {plan_path}")
        
        # 3. Story Generation
        logging.info("Step 3: Generating Final Story...")
        story = generate_story(plan, llm_client, story_prompts, story_config)
        
        story_path = story_config['output_path']
        story.save(story_path)
        logging.info(f"Final story saved to {story_path}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
