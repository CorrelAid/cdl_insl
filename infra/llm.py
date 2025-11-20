"""Modal deployment for Mistral model inference using vLLM."""

import modal

# Create a container image with vLLM and necessary dependencies
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "huggingface-hub==0.36.0",
    )
)

# Configure model - using a smaller Mistral model for efficiency
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "main"

# GPU configuration options (choose based on your needs):
# - "T4" - Budget option, slower 16GB
# - "A10G" - Good balance
# Nvidia A10 $1.10 / h
# Nvidia L4 $0.80 / h
# Nvidia T4 $0.59 / h
GPU_CONFIG = "A10G"

# Create a Volume for caching model weights
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App("insl-mistral-inference")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=3600,  # 1 hour timeout
    scaledown_window=1200,
    secrets=[modal.Secret.from_name("vllm-api-key")],
)
@modal.web_server(port=8000, startup_timeout=180, requires_proxy_auth=True)
def serve():
    """Serve Mistral model using vLLM with built-in API key authentication."""
    import subprocess
    import os

    # Get API key from Modal secret
    api_key = os.getenv("VLLM_API_KEY")

    cmd = [
        "vllm", "serve",
        MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "auto",
        "--max-model-len", "8192",  # Context length
        "--api-key", api_key,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(
    image=vllm_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model():
    """Pre-download the model to the cache volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],  # Exclude unnecessary files
    )


@app.local_entrypoint()
def main():
    """Deploy and get the inference server URL."""
    # First download the model
    print("Downloading model to cache...")
    download_model.remote()

    # Get the server URL
    url = serve.web_url
    print(f"\n✅ Mistral inference server deployed!")
    print(f"🌐 Server URL: {url}")
    print("\n🔐 This endpoint uses vLLM API key authentication.")
    print("   Set VLLM_API_KEY in Modal secret 'vllm-api-key'")
    print("\nExample usage with authentication:")
    print('export VLLM_API_KEY=your-vllm-api-key')
    print(f'curl -X POST "{url}/v1/completions" \\')
    print('  -H "Authorization: Bearer $VLLM_API_KEY" \\')
    print('  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"model": "{MODEL_NAME}", "prompt": "Hello", "max_tokens": 50}}\'')
    print("\n📝 To use with your app, set these in .env:")
    print(f"MODAL_BASE_URL={url}/v1")
    print("VLLM_API_KEY=your-vllm-api-key")


if __name__ == "__main__":
    main()
