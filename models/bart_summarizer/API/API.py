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
        from transformers import pipeline
        import os

        os.environ["HF_HUB_CACHE"] = "/cache"

        self.pipe = pipeline(
            "summarization",
            model="SeifElden2342532/children_educational_summarizer",
            device=0
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def process(self, query: Query):
        result = self.pipe(
            query.text,
            max_length=130,
            min_length=50,
            truncation=True,
            do_sample=False
        )[0]

        return {"summary": result["summary_text"]}
