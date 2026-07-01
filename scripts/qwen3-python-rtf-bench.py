#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5"
DEFAULT_REPO = "/home/flynn/Qwen3-TTS"
DEFAULT_TEXT = (
    "The speaker describes a calm morning in the city, where people walk to work, "
    "shops open their doors, and the first trains leave the station on time."
)


def dtype_from_name(name: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(tts: Any, args: argparse.Namespace, label: str) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    generate_args = tts._merge_generate_kwargs(
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=args.subtalker_dosample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        max_new_tokens=args.max_new_tokens,
    )
    input_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])

    sync(args.device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        codes_list, _extra = tts.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=args.non_streaming_mode,
            **generate_args,
        )
    sync(args.device)
    generation_seconds = time.perf_counter() - generation_start

    sync(args.device)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        wavs, sample_rate = tts.model.speech_tokenizer.decode(
            [{"audio_codes": codes_list[0]}]
        )
    sync(args.device)
    decode_seconds = time.perf_counter() - decode_start

    wav = wavs[0]
    codes = codes_list[0]
    audio_seconds = len(wav) / sample_rate if sample_rate else 0.0
    inference_seconds = generation_seconds + decode_seconds
    result = {
        "label": label,
        "frames": int(codes.shape[0]),
        "code_groups": int(codes.shape[1]),
        "samples": int(len(wav)),
        "sample_rate": int(sample_rate),
        "audio_seconds": audio_seconds,
        "generation_seconds": generation_seconds,
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "generation_rtf": generation_seconds / audio_seconds if audio_seconds else None,
        "decode_rtf": decode_seconds / audio_seconds if audio_seconds else None,
        "inference_rtf": inference_seconds / audio_seconds if audio_seconds else None,
        "generation_kwargs": generate_args,
        "wav": wav,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Python Qwen3-TTS generation/decode RTF for a configurable prompt."
    )
    parser.add_argument("--repo", default=os.environ.get("QWEN3_TTS_PYTHON_SRC", DEFAULT_REPO))
    parser.add_argument("--model", default=os.environ.get("QWEN3_MODEL_DIR", DEFAULT_MODEL))
    parser.add_argument("--device", default=os.environ.get("QWEN3_PY_DEVICE", "cuda:0"))
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=os.environ.get("QWEN3_PY_DTYPE", "float32"),
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--text", default=os.environ.get("QWEN3_TEXT", DEFAULT_TEXT))
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs to exclude MIOpen/kernel setup from measured RTF.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument(
        "--non-streaming-mode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("python-reference/out/qwen3_python_rtf_bench"),
        help="Directory for metadata.json and the last measured output.wav.",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be >= 0 and --iterations must be > 0")

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    load_start = time.perf_counter()
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    sync(args.device)
    load_seconds = time.perf_counter() - load_start

    warmups = [run_once(tts, args, f"warmup-{index + 1}") for index in range(args.warmup)]
    measured = [run_once(tts, args, f"measured-{index + 1}") for index in range(args.iterations)]

    args.out.mkdir(parents=True, exist_ok=True)
    last = measured[-1]
    sf.write(args.out / "output.wav", last["wav"], last["sample_rate"])
    for result in warmups + measured:
        result.pop("wav")

    means = {
        key: sum(result[key] for result in measured) / len(measured)
        for key in [
            "audio_seconds",
            "generation_seconds",
            "decode_seconds",
            "inference_seconds",
            "generation_rtf",
            "decode_rtf",
            "inference_rtf",
        ]
    }
    metadata = {
        "text": args.text,
        "model": args.model,
        "repo": args.repo,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "speaker": args.speaker,
        "language": args.language,
        "non_streaming_mode": args.non_streaming_mode,
        "load_seconds": load_seconds,
        "warmup": warmups,
        "measured": measured,
        "mean": means,
        "output_wav": str(args.out / "output.wav"),
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
