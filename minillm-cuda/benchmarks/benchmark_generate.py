import time
import statistics

import torch
import transformers

from transformers.generation.streamers import BaseStreamer
from minillm.model import load_model, MODEL_NAME


WARMUP = 2
REPEAT = 5
NEW_TOKENS = 128

PROMPT = "What is CUDA? Explain it briefly."


class TimingStreamer(BaseStreamer):

    def __init__(self):
        self.is_prompt = True
        self.timestamps = []

    def put(self, value):

        if self.is_prompt:
            self.is_prompt = False
            return

        if value.numel() != 1:
            raise RuntimeError(
                "Only single-token generation is supported."
            )

        self.timestamps.append(time.perf_counter())

    def end(self):
        pass


def benchmark_once(model, inputs):

    streamer = TimingStreamer()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            return_dict_in_generate=False,
            streamer=streamer
        )

    torch.cuda.synchronize()

    end = time.perf_counter()

    input_tokens = inputs["input_ids"].shape[1]

    output_tokens = (
        output_ids.shape[1] - input_tokens
    )

    timestamps = streamer.timestamps

    if output_tokens != NEW_TOKENS:
        raise RuntimeError("Unexpected output length.")

    if len(timestamps) != output_tokens:
        raise RuntimeError("Token count mismatch.")

    total_time = end - start

    ttft = timestamps[0] - start

    decode_time = timestamps[-1] - timestamps[0]

    tpot = decode_time / (output_tokens - 1)

    overall_throughput = output_tokens / total_time

    decode_throughput = (
        (output_tokens - 1) / decode_time
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft": ttft * 1000,
        "tpot": tpot * 1000,
        "total_time": total_time,
        "overall_throughput": overall_throughput,
        "decode_throughput": decode_throughput,
        "peak_memory": peak_memory
    }


def main():

    tokenizer, model = load_model()

    messages = [
        {"role": "user", "content": PROMPT}
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

    print("\n========== Benchmark Configuration ==========")

    print("Model:", MODEL_NAME)
    print("Model revision:", getattr(model.config, "_commit_hash", None))
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Dtype:", model.dtype)
    print("Attention backend:", model.config._attn_implementation)

    print("Batch size:", inputs["input_ids"].shape[0])
    print("Input tokens:", inputs["input_ids"].shape[1])
    print("Output tokens:", NEW_TOKENS)
    print("Warmup:", WARMUP)
    print("Repeat:", REPEAT)

    print("=============================================")

    print("\n========== Warmup ==========")

    for i in range(WARMUP):

        benchmark_once(model, inputs)

        print(f"Warmup {i + 1}/{WARMUP} completed")

    print("\n========== Benchmark ==========")

    results = []

    for i in range(REPEAT):

        result = benchmark_once(model, inputs)

        results.append(result)

        print(
            f"Run {i + 1}: "
            f"TTFT = {result['ttft']:.3f} ms, "
            f"TPOT = {result['tpot']:.3f} ms, "
            f"Throughput = "
            f"{result['overall_throughput']:.3f} tokens/s"
        )

    print("\n========== Average Results ==========")

    metrics = [
        ("TTFT", "ttft", "ms"),
        ("TPOT", "tpot", "ms/token"),
        ("Total latency", "total_time", "s"),
        ("Overall throughput", "overall_throughput", "tokens/s"),
        ("Decode throughput", "decode_throughput", "tokens/s")
    ]

    for name, key, unit in metrics:

        average = statistics.mean(
            result[key] for result in results
        )

        std = statistics.stdev(
            result[key] for result in results
        )

        print(
            f"{name:<22}: "
            f"{average:.3f} {unit} "
            f"(std: {std:.3f})"
        )

    peak_memory = max(
        result["peak_memory"] for result in results
    )

    print(f"{'Peak GPU memory':<22}: {peak_memory:.3f} GiB")

    print("=====================================")


if __name__ == "__main__":
    main()