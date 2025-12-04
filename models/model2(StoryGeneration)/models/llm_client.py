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