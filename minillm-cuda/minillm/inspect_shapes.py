import torch

from minillm.model import load_model


def shape_hook(name):
    def hook(module, inputs, output):
        input_shape = tuple(inputs[0].shape)
        output_shape = tuple(output.shape)

        print(f"{name:<24} {str(input_shape):<20} -> {output_shape}")

    return hook


def main():
    tokenizer, model = load_model()

    prompt = "What is CUDA? Explain it briefly."

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    layer = model.model.layers[0]

    modules = {
        "Embedding": model.model.embed_tokens,

        "Input RMSNorm": layer.input_layernorm,

        "Q Projection": layer.self_attn.q_proj,
        "K Projection": layer.self_attn.k_proj,
        "V Projection": layer.self_attn.v_proj,
        "Output Projection": layer.self_attn.o_proj,

        "Post-Attn RMSNorm": layer.post_attention_layernorm,

        "Gate Projection": layer.mlp.gate_proj,
        "Up Projection": layer.mlp.up_proj,
        "Down Projection": layer.mlp.down_proj,

        "Final RMSNorm": model.model.norm,
        "LM Head": model.lm_head
    }

    handles = []

    for name, module in modules.items():
        handle = module.register_forward_hook(
            shape_hook(name)
        )

        handles.append(handle)

    print("\n========== Tensor Shapes ==========")
    print("Input tokens:", inputs["input_ids"].shape[1])

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True
            )

    finally:
        for handle in handles:
            handle.remove()

    print("\nOutput shape:", tuple(output_ids.shape))
    print("===================================")


if __name__ == "__main__":
    main()