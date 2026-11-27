import torch

from minillm.model import load_model


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

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True
        )

    generated_ids = output_ids[0, input_length:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("\n========== Generation Result ==========")
    print("Prompt:", prompt)
    print("Input tokens:", input_length)
    print("Generated tokens:", generated_ids.numel())
    print("Response:", response)
    print("=======================================")


if __name__ == "__main__":
    main()