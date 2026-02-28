import os
import modal

app = modal.App("example-inference")
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]", "huggingface_hub")


@app.cls(gpu="h100", image=image, secrets=[modal.Secret.from_name("huggingface")])
class Model:
    @modal.enter()
    def load_model(self):
        """Runs once when the container starts — loads the model into GPU memory."""
        from huggingface_hub import login
        from transformers import pipeline

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.chatbot = pipeline(
            model="Qwen/Qwen3-1.7B-FP8",
            device_map="cuda",
            model_kwargs={"tie_word_embeddings": False},
            token=hf_token,
        )
        print("✅ Model loaded and ready!")

    @modal.method()
    def chat(self, prompt: str) -> str:
        """Send a prompt, get a response."""
        context = [{"role": "user", "content": prompt}]
        result = self.chatbot(context, max_new_tokens=1024)
        return result[0]["generated_text"][-1]["content"]