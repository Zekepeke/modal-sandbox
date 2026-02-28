import modal
import os

# 1. Define the model and image
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm==0.6.3", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("h100-llama-service")
volume = modal.Volume.from_name("model-weights", create_if_missing=True)

@app.cls(
    gpu="H100", 
    image=image, 
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")], # You'll create this next
    timeout=1200 # 20 mins for the first-time download
)
class Model:
    @modal.enter()
    def load(self):
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        
        # Download weights to Volume if not present
        model_path = f"/data/{MODEL_ID}"
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL_ID, local_dir=model_path)
            volume.commit()

        engine_args = AsyncEngineArgs(model=model_path, tensor_parallel_size=1)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(self, user_prompt: str, rag_context: str):
        from vllm import SamplingParams
        import uuid

        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        prompt = f"<|begin_of_text|>System: Use context: {rag_context}\nUser: {user_prompt}\nAssistant:"
        
        # This streams the response back
        results_generator = self.engine.generate(prompt, sampling_params, str(uuid.uuid4()))
        async for request_output in results_generator:
            yield request_output.outputs[0].text