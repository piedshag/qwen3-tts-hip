use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::generation::DEFAULT_STREAM_LEFT_CONTEXT_FRAMES;
use qwen3_hip_runtime::{
    Error, GenerateOptions, HipTtsEngine, Language, Result, Speaker, StreamOptions,
};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
    let max_frames = parse_usize_arg(args.next(), "max_frames")?.unwrap_or(120);
    let chunk_frames = parse_usize_arg(args.next(), "chunk_frames")?.unwrap_or(6);
    let speaker = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Speaker>())
        .transpose()?
        .unwrap_or(Speaker::Ryan);
    let language = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Language>())
        .transpose()?
        .unwrap_or(Language::English);
    if max_frames == 0 || chunk_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames and chunk_frames must be non-zero".to_string(),
        ));
    }

    let load_start = Instant::now();
    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_frames)?;
    engine.runtime().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let options = GenerateOptions {
        speaker,
        language,
        max_frames,
        decode_audio: false,
        ..GenerateOptions::default()
    };
    let stream_options = StreamOptions::default();
    let mut stream = engine.start_stream(&text, options, stream_options.clone())?;
    println!(
        "HIP stream bench: load_seconds={load_seconds:.6}, max_frames={max_frames}, chunk_frames={chunk_frames}, left_context_frames={DEFAULT_STREAM_LEFT_CONTEXT_FRAMES}"
    );

    let mut chunk_index = 0usize;
    let mut total_samples = 0usize;
    let mut last_seconds = 0.0f64;
    while let Some(chunk) = timed_next_audio(&mut stream, chunk_frames, &mut last_seconds)? {
        chunk_index += 1;
        total_samples += chunk.samples.len();
        let audio_seconds =
            chunk.samples.len() as f64 / qwen3_hip_runtime::generation::SAMPLE_RATE as f64;
        let total_audio_seconds =
            total_samples as f64 / qwen3_hip_runtime::generation::SAMPLE_RATE as f64;
        println!(
            "audio_chunk={chunk_index}, seconds={last_seconds:.6}, frames={}, total_frames={}, samples={}, audio_seconds={audio_seconds:.6}, total_audio_seconds={total_audio_seconds:.6}, rtf={:.6}, ended_by_eos={}",
            chunk.frames,
            chunk.total_frames,
            chunk.samples.len(),
            last_seconds / audio_seconds.max(f64::MIN_POSITIVE),
            chunk.ended_by_eos,
        );
    }

    let mut codes_stream = engine.start_stream(
        &text,
        GenerateOptions {
            speaker,
            language,
            max_frames,
            decode_audio: false,
            ..GenerateOptions::default()
        },
        stream_options,
    )?;
    println!("HIP stream code chunk timings:");
    chunk_index = 0;
    while let Some(chunk) = timed_next_codes(&mut codes_stream, chunk_frames, &mut last_seconds)? {
        chunk_index += 1;
        println!(
            "code_chunk={chunk_index}, seconds={last_seconds:.6}, frames={}, total_frames={}, codes={}, ended_by_eos={}",
            chunk.frames,
            chunk.total_frames,
            chunk.codes.len(),
            chunk.ended_by_eos,
        );
    }
    Ok(())
}

fn timed_next_audio(
    stream: &mut qwen3_hip_runtime::HipTtsStream<'_>,
    chunk_frames: usize,
    seconds: &mut f64,
) -> Result<Option<qwen3_hip_runtime::GeneratedAudioChunk>> {
    let start = Instant::now();
    let chunk = stream.next_audio_chunk(chunk_frames)?;
    *seconds = start.elapsed().as_secs_f64();
    Ok(chunk)
}

fn timed_next_codes(
    stream: &mut qwen3_hip_runtime::HipTtsStream<'_>,
    chunk_frames: usize,
    seconds: &mut f64,
) -> Result<Option<qwen3_hip_runtime::GeneratedCodesChunk>> {
    let start = Instant::now();
    let chunk = stream.next_codes_chunk(chunk_frames)?;
    *seconds = start.elapsed().as_secs_f64();
    Ok(chunk)
}

fn parse_usize_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<usize>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}
