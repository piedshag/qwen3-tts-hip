#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPO = Path(os.environ.get("QWEN3_TTS_PYTHON_SRC", "/home/flynn/Qwen3-TTS"))


def add_repo(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Qwen3-TTS repo not found: {repo}")
    sys.path.insert(0, str(repo))


def dtype_from_name(name: str) -> Any:
    import torch

    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return table[name]


def tensor_to_numpy(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def stats(name: str, value: Any) -> dict[str, Any]:
    import numpy as np

    array = tensor_to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    return {
        "name": name,
        "shape": list(tensor_to_numpy(value).shape),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "first8": [float(x) for x in array[:8]],
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def save_array(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(tensor_to_numpy(value))
    np.save(path, array)
    return stats(path.stem, array)


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "device_map": args.device,
        "dtype": dtype_from_name(args.dtype),
        "attn_implementation": args.attn_implementation,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def model_path(args: argparse.Namespace) -> str:
    if not args.local_files_only:
        return args.model
    from huggingface_hub import snapshot_download

    return snapshot_download(
        args.model,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=True,
    )


def load_tts(args: argparse.Namespace):
    add_repo(args.repo)
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(model_path(args), **model_kwargs(args))


def generation_kwargs(tts, args: argparse.Namespace) -> dict[str, Any]:
    return tts._merge_generate_kwargs(
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


def maybe_sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_custom_voice(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch

    torch.manual_seed(args.seed)
    tts = load_tts(args)
    input_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])
    instruct_ids = [None]
    if args.instruct:
        instruct_ids = [tts._tokenize_texts([tts._build_instruct_text(args.instruct)])[0]]
    generate_args = generation_kwargs(tts, args)

    maybe_sync(args.device)
    with torch.inference_mode():
        codes_list, extra = tts.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=args.non_streaming_mode,
            **generate_args,
        )
        wavs, sample_rate = tts.model.speech_tokenizer.decode([{"audio_codes": codes_list[0]}])
    maybe_sync(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    code_stats = save_array(args.out / "talker_codes.npy", codes_list[0])
    hidden_stats = None
    if isinstance(extra, list) and extra:
        hidden_stats = save_array(args.out / "talker_hidden_states.npy", extra[0])
    wav_stats = save_array(args.out / "waveform.npy", wavs[0])
    sf.write(args.out / "output.wav", wavs[0], sample_rate)
    metadata = {
        "mode": "custom-voice",
        "model": args.model,
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "sample_rate": int(sample_rate),
        "wav": wav_stats,
        "talker_codes": code_stats,
        "talker_hidden_states": hidden_stats,
        "extra_type": type(extra).__name__,
        "generation_kwargs": generate_args,
    }
    save_json(args.out / "metadata.json", metadata)


def run_voice_design(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch

    torch.manual_seed(args.seed)
    tts = load_tts(args)
    input_ids = tts._tokenize_texts([tts._build_assistant_text(args.text)])
    instruct_ids = [None]
    if args.instruct:
        instruct_ids = [tts._tokenize_texts([tts._build_instruct_text(args.instruct)])[0]]

    maybe_sync(args.device)
    with torch.inference_mode():
        codes_list, extra = tts.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[args.language],
            non_streaming_mode=args.non_streaming_mode,
            **generation_kwargs(tts, args),
        )
        wavs, sample_rate = tts.model.speech_tokenizer.decode([{"audio_codes": codes_list[0]}])
    maybe_sync(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    code_stats = save_array(args.out / "talker_codes.npy", codes_list[0])
    wav_stats = save_array(args.out / "waveform.npy", wavs[0])
    sf.write(args.out / "output.wav", wavs[0], sample_rate)
    metadata = {
        "mode": "voice-design",
        "model": args.model,
        "text": args.text,
        "language": args.language,
        "instruct": args.instruct,
        "sample_rate": int(sample_rate),
        "wav": wav_stats,
        "talker_codes": code_stats,
        "extra_type": type(extra).__name__,
    }
    save_json(args.out / "metadata.json", metadata)


def run_voice_clone_prompt(args: argparse.Namespace) -> None:
    tts = load_tts(args)
    items = tts.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )
    item = items[0]

    args.out.mkdir(parents=True, exist_ok=True)
    prompt_stats = {
        "mode": "voice-clone-prompt",
        "model": args.model,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "x_vector_only": bool(args.x_vector_only),
        "icl_mode": bool(item.icl_mode),
        "speaker_embedding": save_array(args.out / "speaker_embedding.npy", item.ref_spk_embedding),
    }
    if item.ref_code is not None:
        prompt_stats["ref_code"] = save_array(args.out / "ref_code.npy", item.ref_code)
    save_json(args.out / "metadata.json", prompt_stats)


def run_custom_voice_talker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    tts = load_tts(args)
    model = tts.model
    talker = model.talker
    input_id = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0]

    if args.language.lower() not in model.config.talker_config.codec_language_id:
        raise NotImplementedError(f"Language {args.language} not implemented")
    if args.speaker.lower() not in model.config.talker_config.spk_id:
        raise NotImplementedError(f"Speaker {args.speaker} not implemented")

    language_id = model.config.talker_config.codec_language_id[args.language.lower()]
    speaker_id = model.config.talker_config.spk_id[args.speaker.lower()]

    maybe_sync(args.device)
    with torch.inference_mode():
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                    device=talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        speaker_embed = talker.get_input_embeddings()(
            torch.tensor(speaker_id, device=talker.device, dtype=input_id.dtype)
        )
        codec_prefill = talker.get_input_embeddings()(
            torch.tensor(
                [[
                    model.config.talker_config.codec_think_id,
                    model.config.talker_config.codec_think_bos_id,
                    language_id,
                    model.config.talker_config.codec_think_eos_id,
                ]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
        codec_suffix = talker.get_input_embeddings()(
            torch.tensor(
                [[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
        codec_input_embedding = torch.cat(
            [codec_prefill, speaker_embed.view(1, 1, -1), codec_suffix], dim=1
        )

        role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
        codec_hidden = torch.cat(
            (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed), dim=1
        ) + codec_input_embedding[:, :-1]
        first_text = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4]))
        prefill = torch.cat([role, codec_hidden, first_text + codec_input_embedding[:, -1:]], dim=1)
        trailing_text = torch.cat(
            (talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed), dim=1
        )
        attention_mask = torch.ones(prefill.shape[:2], dtype=torch.long, device=prefill.device)
        outputs = talker(
            inputs_embeds=prefill,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            trailing_text_hidden=trailing_text,
            tts_pad_embed=tts_pad_embed,
        )
        logits = outputs.logits[:, -1:, :]
        suppressed_logits = logits.clone()
        suppress_start = model.config.talker_config.vocab_size - 1024
        suppress_end = model.config.talker_config.vocab_size
        eos = model.config.talker_config.codec_eos_token_id
        suppress_ids = [idx for idx in range(suppress_start, suppress_end) if idx != eos]
        suppressed_logits[..., suppress_ids] = -torch.inf
        first_semantic = torch.argmax(suppressed_logits, dim=-1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=torch.cat(
                (outputs.past_hidden, talker.get_input_embeddings()(first_semantic)), dim=1
            ),
            max_new_tokens=model.config.talker_config.num_code_groups - 1,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        first_acoustic = predictor_result.sequences
        first_frame = torch.cat((first_semantic, first_acoustic), dim=-1)
        first_codec_hiddens = torch.cat(
            [talker.get_input_embeddings()(first_semantic)]
            + [
                talker.code_predictor.get_input_embeddings()[idx](first_acoustic[..., idx : idx + 1])
                for idx in range(model.config.talker_config.num_code_groups - 1)
            ],
            dim=1,
        )
        first_step_input = first_codec_hiddens.sum(1, keepdim=True) + trailing_text[:, 0:1]
        rollout_codes = []
        current_semantic = first_semantic
        current_outputs = outputs
        current_step = 0
        forward_first_frame = None
        forward_next_logits = None
        forward_next_suppressed_logits = None
        next_semantic = None
        rollout_logits = []
        rollout_suppressed_logits = []
        rollout_past_hidden = []
        for frame_index in range(args.max_new_tokens):
            step_outputs = talker(
                input_ids=current_semantic,
                past_key_values=current_outputs.past_key_values,
                past_hidden=current_outputs.past_hidden,
                trailing_text_hidden=trailing_text,
                tts_pad_embed=tts_pad_embed,
                generation_step=current_step,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                subtalker_dosample=False,
                subtalker_top_p=1.0,
                subtalker_top_k=50,
                subtalker_temperature=1.0,
            )
            frame = step_outputs.hidden_states[-1]
            rollout_codes.append(frame)
            current_outputs = step_outputs
            current_step = int(step_outputs.generation_step)
            next_logits = step_outputs.logits[:, -1:, :].clone()
            next_logits[..., suppress_ids] = -torch.inf
            rollout_logits.append(step_outputs.logits[:, -1:, :].clone())
            rollout_suppressed_logits.append(next_logits.clone())
            rollout_past_hidden.append(step_outputs.past_hidden.clone())
            current_semantic = torch.argmax(next_logits, dim=-1)
            if frame_index == 0:
                forward_first_frame = frame
                forward_next_logits = step_outputs.logits[:, -1:, :].clone()
                forward_next_suppressed_logits = next_logits.clone()
                next_semantic = current_semantic.clone()
            if int(current_semantic.item()) == eos:
                break
        rollout_codes = torch.cat(rollout_codes, dim=0) if rollout_codes else torch.empty((0, model.config.talker_config.num_code_groups), device=talker.device)
        rollout_logits = torch.cat(rollout_logits, dim=0) if rollout_logits else torch.empty((0, 1, model.config.talker_config.vocab_size), device=talker.device)
        rollout_suppressed_logits = torch.cat(rollout_suppressed_logits, dim=0) if rollout_suppressed_logits else torch.empty((0, 1, model.config.talker_config.vocab_size), device=talker.device)
        rollout_past_hidden = torch.cat(rollout_past_hidden, dim=0) if rollout_past_hidden else torch.empty((0, 1, talker.config.hidden_size), device=talker.device)
        rollout_step_inputs = []
        for frame_index in range(rollout_codes.shape[0]):
            frame = rollout_codes[frame_index : frame_index + 1]
            codec_hiddens = torch.cat(
                [talker.get_input_embeddings()(frame[..., 0:1])]
                + [
                    talker.code_predictor.get_input_embeddings()[idx](frame[..., idx + 1 : idx + 2])
                    for idx in range(model.config.talker_config.num_code_groups - 1)
                ],
                dim=1,
            )
            text_embed = trailing_text[:, frame_index : frame_index + 1] if frame_index < trailing_text.shape[1] else tts_pad_embed
            rollout_step_inputs.append(codec_hiddens.sum(1, keepdim=True) + text_embed)
        rollout_step_inputs = torch.cat(rollout_step_inputs, dim=0) if rollout_step_inputs else torch.empty((0, 1, talker.config.hidden_size), device=talker.device)

    maybe_sync(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    input_ids_np = input_id.detach().cpu().to(torch.float32).numpy()
    content_ids_np = input_id[:, 3:-5].detach().cpu().to(torch.float32).numpy()
    reports: dict[str, Any] = {
        "mode": "custom-voice-talker",
        "model": args.model,
        "text": args.text,
        "language": args.language,
        "speaker": args.speaker,
        "input_ids": save_array(args.out / "input_ids.npy", input_ids_np),
        "content_ids": save_array(args.out / "content_ids.npy", content_ids_np),
        "prefill": save_array(args.out / "prefill.npy", prefill),
        "trailing_text": save_array(args.out / "trailing_text.npy", trailing_text),
        "tts_pad_embed": save_array(args.out / "tts_pad_embed.npy", tts_pad_embed),
        "last_hidden": save_array(args.out / "last_hidden.npy", outputs.past_hidden),
        "logits": save_array(args.out / "logits.npy", logits),
        "suppressed_logits": save_array(args.out / "suppressed_logits.npy", suppressed_logits),
        "first_semantic": save_array(args.out / "first_semantic.npy", first_semantic),
        "first_acoustic": save_array(args.out / "first_acoustic.npy", first_acoustic),
        "first_frame": save_array(args.out / "first_frame.npy", first_frame),
        "first_step_input": save_array(args.out / "first_step_input.npy", first_step_input),
        "forward_first_frame": save_array(args.out / "forward_first_frame.npy", forward_first_frame),
        "forward_next_logits": save_array(args.out / "forward_next_logits.npy", forward_next_logits),
        "forward_next_suppressed_logits": save_array(args.out / "forward_next_suppressed_logits.npy", forward_next_suppressed_logits),
        "next_semantic": save_array(args.out / "next_semantic.npy", next_semantic),
        "rollout_codes": save_array(args.out / "rollout_codes.npy", rollout_codes),
        "rollout_step_inputs": save_array(args.out / "rollout_step_inputs.npy", rollout_step_inputs),
        "rollout_logits": save_array(args.out / "rollout_logits.npy", rollout_logits),
        "rollout_suppressed_logits": save_array(args.out / "rollout_suppressed_logits.npy", rollout_suppressed_logits),
        "rollout_past_hidden": save_array(args.out / "rollout_past_hidden.npy", rollout_past_hidden),
        "token_ids": {
            "language_id": int(language_id),
            "speaker_id": int(speaker_id),
            "codec_eos": int(eos),
        },
    }
    save_json(args.out / "metadata.json", reports)


def run_tokenizer(args: argparse.Namespace) -> None:
    import soundfile as sf

    add_repo(args.repo)
    from qwen_tts import Qwen3TTSTokenizer

    kwargs: dict[str, Any] = {"device_map": args.device}
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    if args.local_files_only:
        kwargs["local_files_only"] = True
    tokenizer = Qwen3TTSTokenizer.from_pretrained(model_path(args), **kwargs)
    encoded = tokenizer.encode(args.audio, sr=args.sr)
    codes = encoded.audio_codes[0]

    args.out.mkdir(parents=True, exist_ok=True)
    code_stats = save_array(args.out / "audio_codes.npy", codes)
    wavs, sample_rate = tokenizer.decode({"audio_codes": codes})
    wav_stats = save_array(args.out / "decoded_waveform.npy", wavs[0])
    sf.write(args.out / "decoded.wav", wavs[0], sample_rate)
    save_json(
        args.out / "metadata.json",
        {
            "mode": "tokenizer",
            "model": args.model,
            "audio": args.audio,
            "sample_rate": int(sample_rate),
            "audio_codes": code_stats,
            "decoded_wav": wav_stats,
        },
    )


def run_codec_stages(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    add_repo(args.repo)
    from qwen_tts import Qwen3TTSTokenizer

    kwargs: dict[str, Any] = {
        "device_map": args.device,
        "dtype": dtype_from_name(args.dtype),
        "attn_implementation": args.attn_implementation,
    }
    if args.local_files_only:
        kwargs["local_files_only"] = True

    tokenizer_path = Path(args.model_dir) / "speech_tokenizer"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), **kwargs)
    decoder = tokenizer.model.decoder
    codes_np = np.load(args.codes).astype(np.int64)
    if codes_np.ndim == 2:
        codes_np = codes_np[None, :, :]
    codes = torch.from_numpy(codes_np).to(tokenizer.device).long().clamp(min=0).transpose(1, 2)

    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {
        "mode": "codec-stages",
        "model_dir": str(args.model_dir),
        "codes": save_array(args.out / "codes_bqt.npy", codes.transpose(1, 2)),
        "stages": {},
    }

    def save_stage(name: str, value: Any) -> None:
        reports["stages"][name] = save_array(args.out / f"{name}.npy", value)

    with torch.inference_mode():
        hidden = decoder.quantizer.decode(codes)
        save_stage("quantized", hidden)

        hidden = decoder.pre_conv(hidden)
        save_stage("pre_conv", hidden)

        hidden = decoder.pre_transformer(inputs_embeds=hidden.transpose(1, 2)).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        save_stage("pre_transformer", hidden)

        for index, blocks in enumerate(decoder.upsample):
            for block_index, block in enumerate(blocks):
                hidden = block(hidden)
                save_stage(f"upsample_{index}_{block_index}", hidden)

        wav = hidden
        for index, block in enumerate(decoder.decoder):
            wav = block(wav)
            save_stage(f"decoder_{index}", wav)

        wav = wav.clamp(min=-1, max=1)
        save_stage("waveform", wav)

    save_json(args.out / "metadata.json", reports)


def deterministic_hidden(length: int, modulus: float, center: float):
    import torch

    return torch.tensor(
        [((idx % int(modulus)) - center) / modulus for idx in range(length)],
        dtype=torch.float32,
    )


def run_code_predictor(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(args.seed)
    tts = load_tts(args)
    predictor = tts.model.talker.code_predictor
    hidden = predictor.config.hidden_size
    num_acoustic = predictor.config.num_code_groups - 1

    maybe_sync(args.device)
    with torch.inference_mode():
        talker_hidden = deterministic_hidden(hidden, 17.0, 15.0).to(
            device=predictor.device,
            dtype=predictor.dtype,
        ).view(1, 1, hidden)
        semantic_embed = deterministic_hidden(hidden, 23.0, 11.0).to(
            device=predictor.device,
            dtype=predictor.dtype,
        ).view(1, 1, hidden)
        result = predictor.generate(
            inputs_embeds=torch.cat((talker_hidden, semantic_embed), dim=1),
            max_new_tokens=num_acoustic,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        tokens = result.sequences.to(torch.float32)
        embeddings = [
            predictor.get_input_embeddings()[idx](result.sequences[..., idx : idx + 1])
            for idx in range(num_acoustic)
        ]
        embedding_sum = torch.stack(embeddings, dim=0).sum(0)

    maybe_sync(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {
        "mode": "code-predictor",
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "hidden": hidden,
        "num_acoustic": num_acoustic,
        "talker_hidden": save_array(args.out / "talker_hidden.npy", talker_hidden),
        "semantic_embed": save_array(args.out / "semantic_embed.npy", semantic_embed),
        "acoustic_tokens": save_array(args.out / "acoustic_tokens.npy", tokens),
        "embedding_sum": save_array(args.out / "embedding_sum.npy", embedding_sum),
    }
    save_json(args.out / "metadata.json", reports)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction)
    parser.add_argument("--subtalker-top-k", type=int)
    parser.add_argument("--subtalker-top-p", type=float)
    parser.add_argument("--subtalker-temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non-streaming-mode", action=argparse.BooleanOptionalAction, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS Python reference artifacts.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    custom = subcommands.add_parser("custom-voice")
    add_common_model_args(custom)
    add_generation_args(custom)
    custom.add_argument("--speaker", default="Ryan")
    custom.set_defaults(func=run_custom_voice)

    custom_talker = subcommands.add_parser("custom-voice-talker")
    add_common_model_args(custom_talker)
    add_generation_args(custom_talker)
    custom_talker.add_argument("--speaker", default="Ryan")
    custom_talker.set_defaults(func=run_custom_voice_talker)

    design = subcommands.add_parser("voice-design")
    add_common_model_args(design)
    add_generation_args(design)
    design.set_defaults(func=run_voice_design)

    clone = subcommands.add_parser("voice-clone-prompt")
    add_common_model_args(clone)
    clone.add_argument("--ref-audio", required=True)
    clone.add_argument("--ref-text", default="")
    clone.add_argument("--x-vector-only", action="store_true")
    clone.set_defaults(func=run_voice_clone_prompt)

    tokenizer = subcommands.add_parser("tokenizer")
    tokenizer.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    tokenizer.add_argument("--model", required=True)
    tokenizer.add_argument("--device", default="cpu")
    tokenizer.add_argument("--cache-dir", type=Path)
    tokenizer.add_argument("--local-files-only", action="store_true")
    tokenizer.add_argument("--audio", required=True)
    tokenizer.add_argument("--sr", type=int)
    tokenizer.add_argument("--out", type=Path, required=True)
    tokenizer.set_defaults(func=run_tokenizer)

    stages = subcommands.add_parser("codec-stages")
    stages.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    stages.add_argument("--model-dir", type=Path, required=True)
    stages.add_argument("--codes", type=Path, required=True)
    stages.add_argument("--device", default="cpu")
    stages.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    stages.add_argument("--attn-implementation", default="eager")
    stages.add_argument("--local-files-only", action="store_true")
    stages.add_argument("--out", type=Path, required=True)
    stages.set_defaults(func=run_codec_stages)

    code_predictor = subcommands.add_parser("code-predictor")
    add_common_model_args(code_predictor)
    code_predictor.add_argument("--seed", type=int, default=0)
    code_predictor.set_defaults(func=run_code_predictor)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
