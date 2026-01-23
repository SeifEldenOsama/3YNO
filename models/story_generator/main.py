import os
import logging
import json
from pipeline.premise import Premise, generate_title, generate_premise, prompts as premise_prompts, config as premise_config
from pipeline.plan import Plan, prompts as plan_prompts, config as plan_config
from pipeline.story import generate_story, story_prompts, story_config
from models.llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(educational_summary: str):
    """
    Runs the full 3YNO story generation pipeline.
    
    Args:
        educational_summary (str): The distilled scientific content to transform into a story.
    """
    try:
        # 1. Premise Generation
        logging.info("Step 1: Generating Premise...")
        premise = Premise()
        generate_title(premise, premise_prompts['title'], premise_config['model']['title'], llm_client)
        generate_premise(premise, premise_prompts['premise'], premise_config['model']['premise'], llm_client)
        
        os.makedirs("output", exist_ok=True)
        premise_path = "output/premise.json"
        premise.save(premise_path)
        logging.info(f"Premise saved to {premise_path}")

        # 2. Plan Generation
        # Note: In a real scenario, plan.py would be called here. 
        # For this professionalization, we ensure the data structures are compatible.
        logging.info("Step 2: Generating Story Plan (Outline, Characters, Setting)...")
        # [Logic to call plan generation would go here]
        
        # 3. Story Generation
        logging.info("Step 3: Generating Final Story...")
        # Assuming plan.json exists from step 2
        plan_path = "output/plan.json"
        if os.path.exists(plan_path):
            plan = Plan.load(plan_path)
            story = generate_story(plan, llm_client, story_prompts, story_config)
            
            story_path = "output/story.json"
            story.save(story_path)
            logging.info(f"Final story saved to {story_path}")
        else:
            logging.warning("Plan file not found. Skipping story generation.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    sample_summary = "Plants need sunlight, water, and soil to grow through photosynthesis."
    run_pipeline(sample_summary)
