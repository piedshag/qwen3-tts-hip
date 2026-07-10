use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::generation::{
    DEFAULT_STREAM_LEFT_CONTEXT_FRAMES, DEFAULT_TEXT_LOOKAHEAD_TOKENS, SAMPLE_RATE,
};
use qwen3_hip_runtime::{
    Error, GenerateOptions, GeneratedSpeech, HipTtsEngine, Language, Result, Speaker,
    StreamOptions, VoiceClonePrompt,
};

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_MAX_CACHE_FRAMES: usize = 240;
const DEFAULT_STREAM_CHUNK_FRAMES: usize = 6;
const MAX_REQUEST_BYTES: usize = 64 * 1024;

struct ServerState {
    engine: HipTtsEngine,
    voice_clone_prompt: Option<VoiceClonePrompt>,
    load_seconds: f64,
    max_cache_frames: usize,
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

struct GenerationRequest {
    text: String,
    options: GenerateOptions,
    stream_options: StreamOptions,
    wav_gain: f32,
    stream_chunk_frames: usize,
    use_voice_clone: bool,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).ok_or_else(|| {
        Error::InvalidInput(
            "usage: cargo run --release --bin tts-server -- <model_dir> [bind_addr] [max_cache_frames] [voice_clone_prompt_json]"
                .to_string(),
        )
    })?;
    let bind = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| DEFAULT_BIND.to_string());
    let max_cache_frames =
        parse_usize_arg(args.next(), "max_cache_frames")?.unwrap_or(DEFAULT_MAX_CACHE_FRAMES);
    let voice_clone_prompt_path = args.next().map(PathBuf::from);
    if max_cache_frames == 0 {
        return Err(Error::InvalidInput(
            "max_cache_frames must be non-zero".to_string(),
        ));
    }
    let voice_clone_prompt = voice_clone_prompt_path
        .as_deref()
        .map(VoiceClonePrompt::from_json)
        .transpose()?;

    let load_start = Instant::now();
    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_cache_frames)?;
    engine.runtime().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let state = ServerState {
        engine,
        voice_clone_prompt,
        load_seconds,
        max_cache_frames,
    };

    let listener = TcpListener::bind(&bind).map_err(io_error)?;
    println!(
        "TTS server listening on http://{bind}/, model_dir={}, load_seconds={load_seconds:.6}, max_cache_frames={max_cache_frames}, voice_clone_prompt={}",
        model_dir.display(),
        voice_clone_prompt_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = handle_connection(&state, &mut stream) {
                    let message = err.to_string();
                    let _ = send_text(&mut stream, 500, "Internal Server Error", &message);
                    eprintln!("request failed: {message}");
                }
            }
            Err(err) => eprintln!("accept failed: {err}"),
        }
    }
    Ok(())
}

fn handle_connection(state: &ServerState, stream: &mut TcpStream) -> Result<()> {
    let request = read_request(stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => send_html(stream, &index_html(state)),
        ("GET", "/health") => send_text(stream, 200, "OK", "ok\n"),
        ("GET", "/favicon.ico") => send_response(stream, 204, "No Content", &[], &[]),
        ("POST", "/api/generate") => handle_generate(state, stream, &request),
        ("POST", "/api/stream") => handle_stream(state, stream, &request),
        _ => send_text(stream, 404, "Not Found", "not found\n"),
    }
}

fn handle_generate(state: &ServerState, stream: &mut TcpStream, request: &Request) -> Result<()> {
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");
    if !content_type.starts_with("application/x-www-form-urlencoded") {
        return send_text(
            stream,
            415,
            "Unsupported Media Type",
            "expected application/x-www-form-urlencoded\n",
        );
    }

    let Some(request) = parse_generation_request_or_bad_request(state, stream, request)? else {
        return Ok(());
    };
    let text = request.text.as_str();
    let options = request.options;

    let generation_start = Instant::now();
    let generated = if request.use_voice_clone {
        let prompt = state.voice_clone_prompt.as_ref().ok_or_else(|| {
            Error::InvalidInput("server was not started with a voice clone prompt".to_string())
        })?;
        state
            .engine
            .generate_voice_clone_codes(text, prompt, options.clone())?
    } else {
        state.engine.generate_codes(text, options.clone())?
    };
    let generation_seconds = generation_start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    let samples = state.engine.decode_codes(&generated.codes)?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let wav_start = Instant::now();
    let speech = GeneratedSpeech {
        text: text.to_string(),
        codes: generated.codes,
        frames: generated.frames,
        ended_by_eos: generated.ended_by_eos,
        samples,
        sample_rate: SAMPLE_RATE,
    };
    let wav = speech.to_wav_bytes(request.wav_gain)?;
    let wav_seconds = wav_start.elapsed().as_secs_f64();
    let time_to_first_audio_seconds = generation_start.elapsed().as_secs_f64();

    let audio_seconds = speech.audio_seconds();
    let inference_seconds = generation_seconds + decode_seconds;
    let generation_rtf = rtf(generation_seconds, audio_seconds);
    let decode_rtf = rtf(decode_seconds, audio_seconds);
    let inference_rtf = rtf(inference_seconds, audio_seconds);

    println!(
        "generated: voice_clone={}, frames={}, samples={}, ended_by_eos={}, time_to_first_audio_seconds={time_to_first_audio_seconds:.6}, generation_seconds={generation_seconds:.6}, decode_seconds={decode_seconds:.6}, wav_seconds={wav_seconds:.6}, audio_seconds={audio_seconds:.6}, inference_rtf={inference_rtf:.6}",
        request.use_voice_clone,
        speech.frames,
        speech.samples.len(),
        speech.ended_by_eos,
    );

    let headers = vec![
        ("Content-Type", "audio/wav".to_string()),
        ("Cache-Control", "no-store".to_string()),
        ("X-TTS-Load-Seconds", format_seconds(state.load_seconds)),
        (
            "X-TTS-Generation-Seconds",
            format_seconds(generation_seconds),
        ),
        ("X-TTS-Decode-Seconds", format_seconds(decode_seconds)),
        ("X-TTS-Wav-Seconds", format_seconds(wav_seconds)),
        (
            "X-TTS-Time-To-First-Audio-Seconds",
            format_seconds(time_to_first_audio_seconds),
        ),
        ("X-TTS-Inference-Seconds", format_seconds(inference_seconds)),
        ("X-TTS-Audio-Seconds", format_seconds(audio_seconds)),
        ("X-TTS-Generation-RTF", format_seconds(generation_rtf)),
        ("X-TTS-Decode-RTF", format_seconds(decode_rtf)),
        ("X-TTS-Inference-RTF", format_seconds(inference_rtf)),
        ("X-TTS-Frames", speech.frames.to_string()),
        ("X-TTS-Samples", speech.samples.len().to_string()),
        ("X-TTS-Ended-By-EOS", speech.ended_by_eos.to_string()),
        ("X-TTS-Voice-Clone", request.use_voice_clone.to_string()),
        (
            "X-TTS-Repetition-Penalty",
            format_seconds(options.repetition_penalty as f64),
        ),
        ("X-TTS-Do-Sample", options.do_sample.to_string()),
        ("X-TTS-Top-K", options.top_k.to_string()),
        ("X-TTS-Top-P", format_seconds(options.top_p as f64)),
        (
            "X-TTS-Temperature",
            format_seconds(options.temperature as f64),
        ),
        (
            "X-TTS-Subtalker-Do-Sample",
            options.subtalker_dosample.to_string(),
        ),
        ("X-TTS-Subtalker-Top-K", options.subtalker_top_k.to_string()),
        (
            "X-TTS-Subtalker-Top-P",
            format_seconds(options.subtalker_top_p as f64),
        ),
        (
            "X-TTS-Subtalker-Temperature",
            format_seconds(options.subtalker_temperature as f64),
        ),
        ("X-TTS-Seed", options.seed.to_string()),
    ];
    send_response(stream, 200, "OK", &headers, &wav)
}

fn handle_stream(state: &ServerState, stream: &mut TcpStream, request: &Request) -> Result<()> {
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");
    if !content_type.starts_with("application/x-www-form-urlencoded") {
        return send_text(
            stream,
            415,
            "Unsupported Media Type",
            "expected application/x-www-form-urlencoded\n",
        );
    }

    let Some(request) = parse_generation_request_or_bad_request(state, stream, request)? else {
        return Ok(());
    };
    let start = Instant::now();
    let mut tts_stream = if request.use_voice_clone {
        let prompt = state.voice_clone_prompt.as_ref().ok_or_else(|| {
            Error::InvalidInput("server was not started with a voice clone prompt".to_string())
        })?;
        state.engine.start_voice_clone_stream(
            &request.text,
            prompt,
            request.options.clone(),
            request.stream_options.clone(),
        )?
    } else {
        state.engine.start_stream(
            &request.text,
            request.options.clone(),
            request.stream_options.clone(),
        )?
    };
    send_chunked_headers(
        stream,
        200,
        "OK",
        &[
            ("Content-Type", "application/octet-stream".to_string()),
            ("Cache-Control", "no-store".to_string()),
            ("X-TTS-Sample-Rate", SAMPLE_RATE.to_string()),
            ("X-TTS-Sample-Format", "s16le".to_string()),
            (
                "X-TTS-Chunk-Frames",
                request.stream_chunk_frames.to_string(),
            ),
            (
                "X-TTS-Text-Lookahead-Tokens",
                request.stream_options.text_lookahead_tokens.to_string(),
            ),
            (
                "X-TTS-Left-Context-Frames",
                request.stream_options.left_context_frames.to_string(),
            ),
            ("X-TTS-Voice-Clone", request.use_voice_clone.to_string()),
        ],
    )?;

    let mut total_samples = 0usize;
    let mut total_frames = 0usize;
    let mut ended_by_eos = false;
    let mut time_to_first_audio_seconds = None;
    while let Some(chunk) = tts_stream.next_audio_chunk(request.stream_chunk_frames)? {
        if time_to_first_audio_seconds.is_none() && !chunk.samples.is_empty() {
            time_to_first_audio_seconds = Some(start.elapsed().as_secs_f64());
        }
        total_samples += chunk.samples.len();
        total_frames = chunk.total_frames;
        ended_by_eos = chunk.ended_by_eos;
        let bytes = samples_to_pcm16_bytes(&chunk.samples, request.wav_gain);
        write_chunk(stream, &bytes)?;
    }
    finish_chunked_response(stream)?;

    let seconds = start.elapsed().as_secs_f64();
    let audio_seconds = total_samples as f64 / SAMPLE_RATE as f64;
    println!(
        "streamed: voice_clone={}, frames={total_frames}, samples={total_samples}, ended_by_eos={ended_by_eos}, time_to_first_audio_seconds={}, seconds={seconds:.6}, audio_seconds={audio_seconds:.6}, rtf={:.6}",
        request.use_voice_clone,
        time_to_first_audio_seconds
            .map(format_seconds)
            .unwrap_or_else(|| "n/a".to_string()),
        rtf(seconds, audio_seconds)
    );
    Ok(())
}

fn parse_generation_request(state: &ServerState, request: &Request) -> Result<GenerationRequest> {
    let form = parse_form(&request.body)?;
    let text = form
        .get("text")
        .map(String::as_str)
        .unwrap_or("She said she would be here by noon.")
        .trim()
        .to_string();
    if text.is_empty() {
        return Err(Error::InvalidInput("text must not be empty".to_string()));
    }
    let max_frames = form
        .get("max_frames")
        .map(|value| parse_usize(value, "max_frames"))
        .transpose()?
        .unwrap_or(120);
    if max_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames must be non-zero".to_string(),
        ));
    }
    let speaker = form
        .get("speaker")
        .map(|value| value.parse::<Speaker>())
        .transpose()?
        .unwrap_or(Speaker::Ryan);
    let language = form
        .get("language")
        .map(|value| value.parse::<Language>())
        .transpose()?
        .unwrap_or(Language::English);
    let wav_gain = form
        .get("wav_gain")
        .map(|value| parse_f32(value, "wav_gain"))
        .transpose()?
        .unwrap_or(1.0);
    let repetition_penalty = form
        .get("repetition_penalty")
        .map(|value| parse_f32(value, "repetition_penalty"))
        .transpose()?
        .unwrap_or(1.05);
    if repetition_penalty <= 0.0 {
        return Err(Error::InvalidInput(
            "repetition_penalty must be positive".to_string(),
        ));
    }
    let do_sample = form
        .get("do_sample")
        .map(|value| parse_bool(value, "do_sample"))
        .transpose()?
        .unwrap_or(true);
    let top_k = form
        .get("top_k")
        .map(|value| parse_usize(value, "top_k"))
        .transpose()?
        .unwrap_or(50);
    let top_p = form
        .get("top_p")
        .map(|value| parse_f32(value, "top_p"))
        .transpose()?
        .unwrap_or(1.0);
    let temperature = form
        .get("temperature")
        .map(|value| parse_f32(value, "temperature"))
        .transpose()?
        .unwrap_or(0.9);
    let subtalker_dosample = form
        .get("subtalker_dosample")
        .map(|value| parse_bool(value, "subtalker_dosample"))
        .transpose()?
        .unwrap_or(true);
    let subtalker_top_k = form
        .get("subtalker_top_k")
        .map(|value| parse_usize(value, "subtalker_top_k"))
        .transpose()?
        .unwrap_or(50);
    let subtalker_top_p = form
        .get("subtalker_top_p")
        .map(|value| parse_f32(value, "subtalker_top_p"))
        .transpose()?
        .unwrap_or(1.0);
    let subtalker_temperature = form
        .get("subtalker_temperature")
        .map(|value| parse_f32(value, "subtalker_temperature"))
        .transpose()?
        .unwrap_or(0.9);
    let seed = form
        .get("seed")
        .map(|value| parse_u64(value, "seed"))
        .transpose()?
        .unwrap_or(0);
    let text_lookahead_tokens = form
        .get("text_lookahead_tokens")
        .map(|value| parse_usize(value, "text_lookahead_tokens"))
        .transpose()?
        .unwrap_or(DEFAULT_TEXT_LOOKAHEAD_TOKENS);
    if text_lookahead_tokens == 0 {
        return Err(Error::InvalidInput(
            "text_lookahead_tokens must be non-zero".to_string(),
        ));
    }
    let stream_chunk_frames = form
        .get("stream_chunk_frames")
        .map(|value| parse_usize(value, "stream_chunk_frames"))
        .transpose()?
        .unwrap_or(DEFAULT_STREAM_CHUNK_FRAMES);
    if stream_chunk_frames == 0 {
        return Err(Error::InvalidInput(
            "stream_chunk_frames must be non-zero".to_string(),
        ));
    }
    let left_context_frames = form
        .get("left_context_frames")
        .map(|value| parse_usize(value, "left_context_frames"))
        .transpose()?
        .unwrap_or(DEFAULT_STREAM_LEFT_CONTEXT_FRAMES);
    let use_voice_clone = form
        .get("use_voice_clone")
        .map(|value| parse_bool(value, "use_voice_clone"))
        .transpose()?
        .unwrap_or(false);
    if use_voice_clone && state.voice_clone_prompt.is_none() {
        return Err(Error::InvalidInput(
            "voice clone mode requires starting the server with voice_clone_prompt_json"
                .to_string(),
        ));
    }

    Ok(GenerationRequest {
        text,
        options: GenerateOptions {
            speaker,
            language,
            max_frames,
            decode_audio: false,
            do_sample,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            subtalker_dosample,
            subtalker_top_k,
            subtalker_top_p,
            subtalker_temperature,
            seed,
        },
        stream_options: StreamOptions {
            text_lookahead_tokens,
            left_context_frames,
            ..StreamOptions::default()
        },
        wav_gain,
        stream_chunk_frames,
        use_voice_clone,
    })
}

fn parse_generation_request_or_bad_request(
    state: &ServerState,
    stream: &mut TcpStream,
    request: &Request,
) -> Result<Option<GenerationRequest>> {
    match parse_generation_request(state, request) {
        Ok(request) => Ok(Some(request)),
        Err(Error::InvalidInput(message)) => {
            send_text(stream, 400, "Bad Request", &format!("{message}\n"))?;
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

fn send_chunked_headers(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    headers: &[(&str, String)],
) -> Result<()> {
    write!(stream, "HTTP/1.1 {code} {reason}\r\n").map_err(io_error)?;
    write!(stream, "Transfer-Encoding: chunked\r\n").map_err(io_error)?;
    write!(stream, "Connection: close\r\n").map_err(io_error)?;
    for (name, value) in headers {
        write!(stream, "{name}: {}\r\n", sanitize_header_value(value)).map_err(io_error)?;
    }
    write!(stream, "\r\n").map_err(io_error)?;
    stream.flush().map_err(io_error)
}

fn write_chunk(stream: &mut TcpStream, body: &[u8]) -> Result<()> {
    if body.is_empty() {
        return Ok(());
    }
    write!(stream, "{:x}\r\n", body.len()).map_err(io_error)?;
    stream.write_all(body).map_err(io_error)?;
    write!(stream, "\r\n").map_err(io_error)?;
    stream.flush().map_err(io_error)
}

fn finish_chunked_response(stream: &mut TcpStream) -> Result<()> {
    write!(stream, "0\r\n\r\n").map_err(io_error)?;
    stream.flush().map_err(io_error)
}

fn samples_to_pcm16_bytes(samples: &[f32], gain: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let value = (sample * gain).clamp(-1.0, 1.0);
        bytes.extend_from_slice(&((value * i16::MAX as f32) as i16).to_le_bytes());
    }
    bytes
}

fn read_request(stream: &mut TcpStream) -> Result<Request> {
    let mut buffer = Vec::new();
    let mut temp = [0u8; 4096];
    let header_end;
    loop {
        let read = stream.read(&mut temp).map_err(io_error)?;
        if read == 0 {
            return Err(Error::InvalidInput("connection closed".to_string()));
        }
        buffer.extend_from_slice(&temp[..read]);
        if buffer.len() > MAX_REQUEST_BYTES {
            return Err(Error::InvalidInput("request is too large".to_string()));
        }
        if let Some(index) = find_header_end(&buffer) {
            header_end = index;
            break;
        }
    }

    let header_text = std::str::from_utf8(&buffer[..header_end])
        .map_err(|err| Error::InvalidInput(format!("invalid HTTP headers: {err}")))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| Error::InvalidInput("missing request line".to_string()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| Error::InvalidInput("missing method".to_string()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| Error::InvalidInput("missing path".to_string()))?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let content_length = headers
        .get("content-length")
        .map(|value| parse_usize(value, "content-length"))
        .transpose()?
        .unwrap_or(0);
    if content_length > MAX_REQUEST_BYTES {
        return Err(Error::InvalidInput("request body is too large".to_string()));
    }

    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let read = stream.read(&mut temp).map_err(io_error)?;
        if read == 0 {
            return Err(Error::InvalidInput("truncated request body".to_string()));
        }
        buffer.extend_from_slice(&temp[..read]);
        if buffer.len() > body_start + MAX_REQUEST_BYTES {
            return Err(Error::InvalidInput("request is too large".to_string()));
        }
    }
    let body = buffer[body_start..body_start + content_length].to_vec();
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn send_html(stream: &mut TcpStream, html: &str) -> Result<()> {
    send_response(
        stream,
        200,
        "OK",
        &[
            ("Content-Type", "text/html; charset=utf-8".to_string()),
            ("Cache-Control", "no-store".to_string()),
        ],
        html.as_bytes(),
    )
}

fn send_text(stream: &mut TcpStream, code: u16, reason: &str, text: &str) -> Result<()> {
    send_response(
        stream,
        code,
        reason,
        &[("Content-Type", "text/plain; charset=utf-8".to_string())],
        text.as_bytes(),
    )
}

fn send_response(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    headers: &[(&str, String)],
    body: &[u8],
) -> Result<()> {
    let mut response = Vec::new();
    write!(response, "HTTP/1.1 {code} {reason}\r\n").map_err(io_error)?;
    write!(response, "Content-Length: {}\r\n", body.len()).map_err(io_error)?;
    write!(response, "Connection: close\r\n").map_err(io_error)?;
    for (name, value) in headers {
        write!(response, "{name}: {}\r\n", sanitize_header_value(value)).map_err(io_error)?;
    }
    write!(response, "\r\n").map_err(io_error)?;
    response.extend_from_slice(body);
    stream.write_all(&response).map_err(io_error)
}

fn index_html(state: &ServerState) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3 HIP TTS Server</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; background: #101218; color: #f3f0e8; }}
    main {{ max-width: 860px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 7vw, 4.8rem); letter-spacing: -0.06em; }}
    p {{ color: #bdb7aa; line-height: 1.5; }}
    form {{ display: grid; gap: 14px; margin: 28px 0; padding: 20px; border: 1px solid #343848; border-radius: 18px; background: #171a23; }}
    label {{ display: grid; gap: 6px; font-size: 0.9rem; color: #d8d1c3; }}
    textarea, input, select, button {{ font: inherit; border-radius: 12px; border: 1px solid #3b4051; background: #0f1118; color: #f3f0e8; padding: 11px 12px; }}
    textarea {{ min-height: 110px; resize: vertical; }}
    .row {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .wide {{ grid-column: 1 / -1; }}
    .section {{ display: grid; gap: 12px; padding-top: 12px; border-top: 1px solid #2d3240; }}
    .actions {{ display: grid; gap: 12px; }}
    .hidden {{ display: none !important; }}
    button {{ cursor: pointer; border: 0; background: linear-gradient(135deg, #f6c177, #eb6f92); color: #16120d; font-weight: 800; }}
    button:disabled {{ cursor: wait; filter: grayscale(0.7); opacity: 0.75; }}
    audio {{ width: 100%; margin: 8px 0 18px; }}
    pre {{ overflow-x: auto; padding: 16px; border-radius: 14px; background: #0b0d12; border: 1px solid #2d3240; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill {{ padding: 6px 10px; border-radius: 999px; background: #232838; color: #d8d1c3; font-size: 0.85rem; }}
    @media (max-width: 720px) {{ .row {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Qwen3 HIP TTS</h1>
  <p>Small standard-library HTTP server around <code>HipTtsEngine</code>. It returns WAV audio and reports generation timing and RTF from response headers.</p>
  <div class="meta">
    <span class="pill">model load: {load_seconds:.3}s</span>
    <span class="pill">initial cache frames: {max_cache_frames}</span>
    <span class="pill">sample rate: {sample_rate} Hz</span>
    <span class="pill">voice clone prompt: {voice_clone_prompt_status}</span>
  </div>
  <form id="tts-form">
    <label>Text
      <textarea name="text">She said she would be here by noon.</textarea>
    </label>
    <div class="row">
      <label>Output mode
        <select name="output_mode" id="output-mode">
          <option value="wav">Generate WAV</option>
          <option value="stream">Stream PCM</option>
        </select>
      </label>
      <label>Output frames
        <input name="max_frames" type="number" min="1" value="120">
      </label>
      <label id="speaker-field">Speaker
        <select name="speaker">
          <option>Ryan</option><option>Serena</option><option>Vivian</option><option>UncleFu</option><option>Aiden</option><option>OnoAnna</option><option>Sohee</option><option>Eric</option><option>Dylan</option>
        </select>
      </label>
      <label>Voice mode
        <select name="use_voice_clone">
          <option value="false">Built-in speaker</option>
          <option value="true" {voice_clone_disabled}>Voice clone prompt</option>
        </select>
      </label>
      <label>Language
        <select name="language">
          <option>English</option><option>Chinese</option><option>Japanese</option><option>Korean</option><option>German</option><option>French</option><option>Russian</option><option>Portuguese</option><option>Spanish</option><option>Italian</option>
        </select>
      </label>
      <label>Repeat penalty
        <input name="repetition_penalty" type="number" step="0.01" min="0.01" value="1.05">
      </label>
      <label>Output gain
        <input name="wav_gain" type="number" step="0.1" value="1.0">
      </label>
    </div>
    <div class="row">
      <label>Sample semantic
        <select name="do_sample"><option value="true">true</option><option value="false">false</option></select>
      </label>
      <label>Top K
        <input name="top_k" type="number" min="0" value="50">
      </label>
      <label>Top P
        <input name="top_p" type="number" step="0.01" min="0.01" max="1" value="1.0">
      </label>
      <label>Temperature
        <input name="temperature" type="number" step="0.01" min="0.01" value="0.9">
      </label>
      <label>Seed
        <input name="seed" type="number" min="0" value="0">
      </label>
      <label>Sample acoustic
        <select name="subtalker_dosample"><option value="true">true</option><option value="false">false</option></select>
      </label>
      <label>Acoustic Top K
        <input name="subtalker_top_k" type="number" min="0" value="50">
      </label>
      <label>Acoustic Top P
        <input name="subtalker_top_p" type="number" step="0.01" min="0.01" max="1" value="1.0">
      </label>
      <label>Acoustic Temp
        <input name="subtalker_temperature" type="number" step="0.01" min="0.01" value="0.9">
      </label>
      <p class="wide">Defaults match Qwen TTS generation settings.</p>
    </div>
    <div class="section" id="wav-options">
      <p class="wide">Full generation uses all text context for best one-shot quality.</p>
    </div>
    <div class="section hidden" id="stream-options">
      <div class="row">
        <label>Stream chunk frames
          <input name="stream_chunk_frames" type="number" min="1" value="{default_stream_chunk_frames}">
        </label>
        <label>Text lookahead
          <input name="text_lookahead_tokens" type="number" min="1" value="{default_text_lookahead_tokens}">
        </label>
        <label>Left context frames
          <input name="left_context_frames" type="number" min="0" value="{default_left_context_frames}">
        </label>
      </div>
      <p class="wide">Streaming options trade time-to-first-audio against chunk overhead and initial context.</p>
    </div>
    <div class="actions">
      <button id="submit" type="submit">Generate WAV</button>
    </div>
  </form>
  <audio id="audio" controls></audio>
  <pre id="stats">No generation yet.</pre>
</main>
<script>
const form = document.getElementById('tts-form');
const button = document.getElementById('submit');
const audio = document.getElementById('audio');
const stats = document.getElementById('stats');
const speakerField = document.getElementById('speaker-field');
const voiceMode = form.elements.use_voice_clone;
const actionMode = form.elements.output_mode;
const wavOptions = document.getElementById('wav-options');
const streamOptions = document.getElementById('stream-options');
const sampleRate = {sample_rate};
let currentUrl = null;

function setActionMode(mode) {{
  wavOptions.classList.toggle('hidden', mode !== 'wav');
  streamOptions.classList.toggle('hidden', mode !== 'stream');
  button.textContent = mode === 'stream' ? 'Stream PCM' : 'Generate WAV';
}}

function updateVoiceMode() {{
  speakerField.classList.toggle('hidden', voiceMode.value === 'true');
}}

voiceMode.addEventListener('change', updateVoiceMode);
actionMode.addEventListener('change', () => setActionMode(actionMode.value));
updateVoiceMode();
setActionMode(actionMode.value);

const statHeaders = [
  ['load_seconds', 'x-tts-load-seconds'],
  ['generation_seconds', 'x-tts-generation-seconds'],
  ['decode_seconds', 'x-tts-decode-seconds'],
  ['wav_seconds', 'x-tts-wav-seconds'],
  ['time_to_first_audio_seconds', 'x-tts-time-to-first-audio-seconds'],
  ['inference_seconds', 'x-tts-inference-seconds'],
  ['audio_seconds', 'x-tts-audio-seconds'],
  ['generation_rtf', 'x-tts-generation-rtf'],
  ['decode_rtf', 'x-tts-decode-rtf'],
  ['inference_rtf', 'x-tts-inference-rtf'],
  ['frames', 'x-tts-frames'],
  ['samples', 'x-tts-samples'],
  ['ended_by_eos', 'x-tts-ended-by-eos'],
  ['voice_clone', 'x-tts-voice-clone'],
  ['repetition_penalty', 'x-tts-repetition-penalty'],
  ['do_sample', 'x-tts-do-sample'],
  ['top_k', 'x-tts-top-k'],
  ['top_p', 'x-tts-top-p'],
  ['temperature', 'x-tts-temperature'],
  ['subtalker_dosample', 'x-tts-subtalker-do-sample'],
  ['subtalker_top_k', 'x-tts-subtalker-top-k'],
  ['subtalker_top_p', 'x-tts-subtalker-top-p'],
  ['subtalker_temperature', 'x-tts-subtalker-temperature'],
  ['seed', 'x-tts-seed'],
  ['text_lookahead_tokens', 'x-tts-text-lookahead-tokens'],
  ['left_context_frames', 'x-tts-left-context-frames'],
];

async function generateWav() {{
  button.disabled = true;
  stats.textContent = 'Generating...';
  try {{
    const requestStart = performance.now();
    const response = await fetch('/api/generate', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
      body: new URLSearchParams(new FormData(form)),
    }});
    const responseHeadersSeconds = (performance.now() - requestStart) / 1000;
    if (!response.ok) {{
      throw new Error(await response.text());
    }}
    const values = Object.fromEntries(statHeaders.map(([name, header]) => [name, response.headers.get(header)]));
    const blob = await response.blob();
    values.client_response_headers_seconds = responseHeadersSeconds.toFixed(3);
    values.client_wav_loaded_seconds = ((performance.now() - requestStart) / 1000).toFixed(3);
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(blob);
    audio.src = currentUrl;
    stats.textContent = JSON.stringify(values, null, 2);
  }} catch (error) {{
    stats.textContent = String(error.message || error);
  }} finally {{
    button.disabled = false;
  }}
}}

async function streamPcm() {{
  button.disabled = true;
  stats.textContent = 'Starting stream...';
  try {{
    const requestStart = performance.now();
    const response = await fetch('/api/stream', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
      body: new URLSearchParams(new FormData(form)),
    }});
    const responseHeadersSeconds = (performance.now() - requestStart) / 1000;
    if (!response.ok) {{
      throw new Error(await response.text());
    }}
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    await context.resume();
    const reader = response.body.getReader();
    let pending = new Uint8Array(0);
    let nextTime = context.currentTime + 0.08;
    let chunks = 0;
    let samples = 0;
    let firstAudioSeconds = null;
    while (true) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      let bytes = value;
      if (pending.length) {{
        const merged = new Uint8Array(pending.length + value.length);
        merged.set(pending, 0);
        merged.set(value, pending.length);
        bytes = merged;
      }}
      if (bytes.length % 2) {{
        pending = bytes.slice(bytes.length - 1);
        bytes = bytes.slice(0, bytes.length - 1);
      }} else {{
        pending = new Uint8Array(0);
      }}
      const frameCount = bytes.length / 2;
      if (!frameCount) continue;
      if (firstAudioSeconds === null) firstAudioSeconds = (performance.now() - requestStart) / 1000;
      const floats = new Float32Array(frameCount);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < frameCount; i++) {{
        floats[i] = view.getInt16(i * 2, true) / 32768;
      }}
      const buffer = context.createBuffer(1, frameCount, sampleRate);
      buffer.copyToChannel(floats, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      if (nextTime < context.currentTime + 0.02) nextTime = context.currentTime + 0.02;
      source.start(nextTime);
      nextTime += buffer.duration;
      chunks += 1;
      samples += frameCount;
      stats.textContent = JSON.stringify({{
        mode: 'stream',
        chunks,
        samples,
        audio_seconds: (samples / sampleRate).toFixed(3),
        client_response_headers_seconds: responseHeadersSeconds.toFixed(3),
        client_time_to_first_audio_seconds: firstAudioSeconds === null ? null : firstAudioSeconds.toFixed(3),
      }}, null, 2);
    }}
  }} catch (error) {{
    stats.textContent = String(error.message || error);
  }} finally {{
    button.disabled = false;
  }}
}}

form.addEventListener('submit', async (event) => {{
  event.preventDefault();
  if (actionMode.value === 'stream') {{
    await streamPcm();
  }} else {{
    await generateWav();
  }}
}});
</script>
</body>
</html>
"#,
        load_seconds = state.load_seconds,
        max_cache_frames = state.max_cache_frames,
        sample_rate = SAMPLE_RATE,
        default_stream_chunk_frames = DEFAULT_STREAM_CHUNK_FRAMES,
        default_text_lookahead_tokens = DEFAULT_TEXT_LOOKAHEAD_TOKENS,
        default_left_context_frames = DEFAULT_STREAM_LEFT_CONTEXT_FRAMES,
        voice_clone_prompt_status = if state.voice_clone_prompt.is_some() {
            "loaded"
        } else {
            "none"
        },
        voice_clone_disabled = if state.voice_clone_prompt.is_some() {
            ""
        } else {
            "disabled"
        },
    )
}

fn parse_form(body: &[u8]) -> Result<HashMap<String, String>> {
    let body = std::str::from_utf8(body)
        .map_err(|err| Error::InvalidInput(format!("form body is not UTF-8: {err}")))?;
    let mut form = HashMap::new();
    for pair in body.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        form.insert(percent_decode(key)?, percent_decode(value)?);
    }
    Ok(form)
}

fn percent_decode(value: &str) -> Result<String> {
    let mut output = Vec::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let high = hex_value(bytes[index + 1])?;
                let low = hex_value(bytes[index + 2])?;
                output.push((high << 4) | low);
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output)
        .map_err(|err| Error::InvalidInput(format!("form value is not UTF-8: {err}")))
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(Error::InvalidInput("invalid percent escape".to_string())),
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_usize_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<usize>> {
    value
        .map(|value| parse_usize(&value.to_string_lossy(), name))
        .transpose()
}

fn parse_usize(value: &str, name: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
}

fn parse_f32(value: &str, name: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
}

fn parse_u64(value: &str, name: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
}

fn parse_bool(value: &str, name: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(Error::InvalidInput(format!("invalid {name}: {other}"))),
    }
}

fn rtf(seconds: f64, audio_seconds: f64) -> f64 {
    if audio_seconds > 0.0 {
        seconds / audio_seconds
    } else {
        0.0
    }
}

fn format_seconds(value: f64) -> String {
    format!("{value:.6}")
}

fn sanitize_header_value(value: &str) -> String {
    value.replace(['\r', '\n'], " ")
}

fn io_error(err: std::io::Error) -> Error {
    Error::InvalidInput(err.to_string())
}
