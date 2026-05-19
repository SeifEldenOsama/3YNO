import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.inference import GLMOCRInference

def main():
    parser = argparse.ArgumentParser(description="GLM-OCR Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, help="OCR prompt (e.g., 'Text Recognition:')")
    parser.add_argument("--max-tokens", type=int, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    cfg = Config.load()
    infer = GLMOCRInference(cfg)
    
    print(f"Processing image: {args.image}")
    result = infer.run(args.image, prompt=args.prompt, max_new_tokens=args.max_tokens)
    
    print("\n--- OCR Result ---")
    print(result)
    print("------------------")

if __name__ == "__main__":
    main()
