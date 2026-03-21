import modal
from pydantic import BaseModel

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "fastapi[standard]"
    )
)

volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

app = modal.App("summarizer-api")

class Query(BaseModel):
    text: str

@app.cls(
    gpu="L4",
    image=image,
    volumes={"/cache": volume},
    scaledown_window=300
)
class Summarizer:

    @modal.enter()
    def load(self):
        from transformers import AutoTokenizer, BartForConditionalGeneration
        import os

        os.environ["HF_HUB_CACHE"] = "/cache"

        MODEL_NAME = "SeifElden2342532/children_educational_summarizer"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")
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
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                max_length=256,
                early_stopping=True
            )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return {"summary": summary}
