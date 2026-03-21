import modal
from pydantic import BaseModel

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "peft",
        "fastapi[standard]"
    )
)

volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

app = modal.App("summarizer-api")

class Query(BaseModel):
    text:       str
    max_length: int = 256
    num_beams:  int = 4

@app.cls(
    gpu="L4",
    image=image,
    volumes={"/cache": volume},
    scaledown_window=300
)
class Summarizer:

    @modal.enter()
    def load(self):
        import os
        import torch
        from transformers import AutoTokenizer, BartForConditionalGeneration
        from peft import PeftModel

        os.environ["HF_HUB_CACHE"] = "/cache"

        BASE_MODEL  = "facebook/bart-large-cnn"
        LORA_ADAPTER = "SeifElden2342532/children_educational_summarizer"

        self.tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER)

        base        = BartForConditionalGeneration.from_pretrained(BASE_MODEL)
        model       = PeftModel.from_pretrained(base, LORA_ADAPTER)
        self.model  = model.merge_and_unload().to("cuda")
        self.model.eval()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def process(self, query: Query):
        import torch

        inputs = self.tokenizer(
            query.text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                num_beams      = query.num_beams,
                max_length     = query.max_length,
                early_stopping = True,
            )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return {
            "summary":      summary,
            "input_words":  len(query.text.split()),
            "output_words": len(summary.split()),
        }