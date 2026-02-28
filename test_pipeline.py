import os
import modal

app = modal.App("example-inference")
volume = modal.Volume.from_name("model-cache", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("ffmpeg").uv_pip_install(
    "transformers[torch]", "huggingface_hub", "openai-whisper", "accelerate"
)


@app.cls(gpu="h100", image=image, secrets=[modal.Secret.from_name("huggingface")], keep_warm=1, volumes={"/cache": volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        import whisper
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = "/cache/huggingface"
        os.environ["WHISPER_CACHE"] = "/cache/whisper"

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        model_name = "Qwen/Qwen3-1.7B-FP8"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )

        self.whisper_model = whisper.load_model("large", device="cuda", download_root="/cache/whisper")

        print("✅ LLM and Whisper loaded and ready!")

    @modal.method()
    def chat(self, prompt: str) -> str:
        """Send a prompt, get a response."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text using Whisper on GPU."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = self.whisper_model.transcribe(temp_path)

        os.unlink(temp_path)
        return result["text"].strip()