import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map={"": 0}
    )

    model.eval()

    return tokenizer, model


def main():
    tokenizer, model = load_model()

    config = model.config

    print("\n========== Model Information ==========")
    print("Model:", MODEL_NAME)
    print("Model type:", config.model_type)
    print("Device:", model.device)
    print("Dtype:", model.dtype)
    print("Layers:", config.num_hidden_layers)
    print("Hidden size:", config.hidden_size)
    print("Intermediate size:", config.intermediate_size)
    print("Query heads:", config.num_attention_heads)
    print("KV heads:", config.num_key_value_heads)
    print("Head dimension:",
          config.hidden_size // config.num_attention_heads)
    print("Vocabulary size:", config.vocab_size)
    print("RMSNorm epsilon:", config.rms_norm_eps)
    print("Attention backend:", config._attn_implementation)

    params = sum(p.numel() for p in model.parameters())

    print("Parameters:", params)

    memory = torch.cuda.memory_allocated() / (1024 ** 3)

    print(f"GPU memory allocated: {memory:.3f} GiB")
    print("=======================================")


if __name__ == "__main__":
    main()