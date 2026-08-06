use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use clap::Parser;
use ndarray::{Array1, Array2};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MODEL_ONNX_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx";
const MODEL_JSON_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json";
const MODEL_ONNX: &str = "zh_CN-huayan-x_low.onnx";
const MODEL_JSON: &str = "zh_CN-huayan-x_low.onnx.json";

const MODEL_EN_ONNX_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx";
const MODEL_EN_JSON_URL: &str = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json";
const MODEL_EN_ONNX: &str = "en_US-lessac-medium.onnx";
const MODEL_EN_JSON: &str = "en_US-lessac-medium.onnx.json";

const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

// Softer voice: lower noise (cleaner/calmer), slower pace, steadier pitch.
const SOFT_NOISE_SCALE: f32 = 0.2;
const SOFT_LENGTH_SCALE: f32 = 1.15;
const SOFT_NOISE_W: f32 = 0.7;

#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}

#[derive(Deserialize)]
struct ESpeakConfig {
    voice: String,
}

#[derive(Deserialize, Clone)]
struct InferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Deserialize)]
struct ModelConfig {
    audio: AudioConfig,
    espeak: ESpeakConfig,
    inference: InferenceConfig,
    num_speakers: u32,
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

#[derive(Parser)]
#[command(name = "noise", about = "Agent notification TTS in Chinese")]
enum Cli {
    #[command(about = "Say \"任务完成\" (task complete)")]
    Done,
    #[command(about = "Say \"等待确认\" (awaiting confirmation)")]
    Confirm,
    #[command(about = "Say \"需要确认方案\" (plan confirmation needed)")]
    Plan,
    #[command(about = "Speak arbitrary text (auto-picks Chinese or English voice)")]
    Say { text: String },
}

fn models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("noise")
        .join("models")
}

fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("noise")
}

fn model_onnx_path() -> PathBuf {
    models_dir().join(MODEL_ONNX)
}

fn model_json_path() -> PathBuf {
    models_dir().join(MODEL_JSON)
}

fn model_en_onnx_path() -> PathBuf {
    models_dir().join(MODEL_EN_ONNX)
}

fn model_en_json_path() -> PathBuf {
    models_dir().join(MODEL_EN_JSON)
}

fn is_cjk(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{3400}'..='\u{4dbf}' |
            '\u{4e00}'..='\u{9fff}' |
            '\u{f900}'..='\u{faff}')
    })
}

fn model_paths(text: &str) -> (PathBuf, PathBuf, String) {
    if is_cjk(text) {
        (model_onnx_path(), model_json_path(), MODEL_ONNX.to_string())
    } else {
        (
            model_en_onnx_path(),
            model_en_json_path(),
            MODEL_EN_ONNX.to_string(),
        )
    }
}

fn download_model(
    onnx_path: &std::path::Path,
    json_path: &std::path::Path,
    onnx_url: &str,
    json_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;

    if !onnx_path.exists() {
        let name = onnx_url.rsplit('/').next().unwrap_or("model");
        eprintln!("Downloading {name}...");
        let resp = reqwest::blocking::get(onnx_url)?;
        let data = resp.bytes()?;
        std::fs::write(onnx_path, &data)?;
    }

    if !json_path.exists() {
        let resp = reqwest::blocking::get(json_url)?;
        let data = resp.bytes()?;
        std::fs::write(json_path, &data)?;
    }

    Ok(())
}

fn load_config(json_path: &std::path::Path) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(json_path)?;
    let config: ModelConfig = serde_json::from_str(&data)?;
    Ok(config)
}

fn phonemize(text: &str, voice: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("espeak-ng")
        .args(["-v", voice, "-x", "-q", "--ipa", text])
        .output()?;
    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    let stripped = strip_lang_switches(&raw);
    Ok(stripped.trim().to_string())
}

fn strip_lang_switches(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: usize = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn phonemes_to_ids(config: &ModelConfig, phonemes: &str) -> Vec<i64> {
    let map = &config.phoneme_id_map;
    let pad_id = *map.get(&PAD).and_then(|v| v.first()).unwrap_or(&0);
    let bos_id = *map.get(&BOS).and_then(|v| v.first()).unwrap_or(&0);
    let eos_id = *map.get(&EOS).and_then(|v| v.first()).unwrap_or(&0);

    let mut ids = Vec::with_capacity(phonemes.len() * 2 + 2);
    ids.push(bos_id);
    for ch in phonemes.chars() {
        if let Some(id) = map.get(&ch).and_then(|v| v.first()) {
            ids.push(*id);
            ids.push(pad_id);
        }
    }
    ids.push(eos_id);
    ids
}

fn infer(
    session: &mut Session,
    config: &ModelConfig,
    phonemes: &str,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: i64,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ids = phonemes_to_ids(config, phonemes);
    let input_len = ids.len();

    let input = Array2::<i64>::from_shape_vec((1, input_len), ids)?;
    let input_lengths = Array1::<i64>::from_iter([input_len as i64]);
    let scales = Array1::<f32>::from_iter([noise_scale, length_scale, noise_w]);

    let input_t =
        Tensor::from_array(([1i64, input_len as i64], input.into_raw_vec_and_offset().0))?;
    let lengths_t = Tensor::from_array(([1i64], input_lengths.into_raw_vec_and_offset().0))?;
    let scales_t = Tensor::from_array(([3i64], scales.into_raw_vec_and_offset().0))?;

    let outputs = if config.num_speakers > 1 {
        let sid = Array1::<i64>::from_iter([speaker_id]);
        let sid_t = Tensor::from_array(([1i64], sid.into_raw_vec_and_offset().0))?;
        session.run(ort::inputs![input_t, lengths_t, scales_t, sid_t])?
    } else {
        session.run(ort::inputs![input_t, lengths_t, scales_t])?
    };

    let (_shape, audio) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(audio.to_vec())
}

fn synthesize(
    text: &str,
    onnx_path: &std::path::Path,
    json_path: &std::path::Path,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let config = load_config(json_path)?;
    let phonemes = phonemize(text, &config.espeak.voice)?;

    let mut session = Session::builder()?.commit_from_file(onnx_path)?;

    let samples = infer(
        &mut session,
        &config,
        &phonemes,
        SOFT_NOISE_SCALE,
        SOFT_LENGTH_SCALE,
        SOFT_NOISE_W,
        0,
    )?;

    Ok((samples, config.audio.sample_rate))
}

fn cache_path(text: &str, model: &str) -> PathBuf {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    h.update(model.as_bytes());
    h.update(SOFT_NOISE_SCALE.to_le_bytes());
    h.update(SOFT_LENGTH_SCALE.to_le_bytes());
    h.update(SOFT_NOISE_W.to_le_bytes());
    let hash = hex::encode(h.finalize());
    cache_dir().join(format!("{}.wav", &hash[..16]))
}

fn write_wav(w: &mut impl Write, samples: &[f32], sample_rate: u32) {
    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    let data_len = (samples_i16.len() * 2) as u32;
    let channels: u16 = 1;
    let byte_rate = sample_rate * channels as u32 * 2;

    w.write_all(b"RIFF").unwrap();
    w.write_all(&(36u32 + data_len).to_le_bytes()).unwrap();
    w.write_all(b"WAVEfmt ").unwrap();
    w.write_all(&16u32.to_le_bytes()).unwrap();
    w.write_all(&1u16.to_le_bytes()).unwrap();
    w.write_all(&channels.to_le_bytes()).unwrap();
    w.write_all(&sample_rate.to_le_bytes()).unwrap();
    w.write_all(&byte_rate.to_le_bytes()).unwrap();
    w.write_all(&(channels * 2).to_le_bytes()).unwrap();
    w.write_all(&16u16.to_le_bytes()).unwrap();
    w.write_all(b"data").unwrap();
    w.write_all(&data_len.to_le_bytes()).unwrap();
    for &s in &samples_i16 {
        w.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn speak(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (onnx, json, model_name) = model_paths(text);
    let path = cache_path(text, &model_name);

    if !path.exists() {
        eprintln!("Synthesizing: {text}");
        let (samples, sample_rate) = synthesize(text, &onnx, &json)?;
        std::fs::create_dir_all(cache_dir())?;
        let mut f = std::fs::File::create(&path)?;
        write_wav(&mut f, &samples, sample_rate);
    }

    let file = std::fs::File::open(&path)?;
    let (_stream, handle) = rodio::OutputStream::try_default()?;
    let sink = rodio::Sink::try_new(&handle)?;
    sink.append(rodio::Decoder::new(std::io::BufReader::new(file))?);
    sink.sleep_until_end();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if !model_onnx_path().exists() || !model_json_path().exists() {
        download_model(
            &model_onnx_path(),
            &model_json_path(),
            MODEL_ONNX_URL,
            MODEL_JSON_URL,
        )?;
    }
    if !model_en_onnx_path().exists() || !model_en_json_path().exists() {
        download_model(
            &model_en_onnx_path(),
            &model_en_json_path(),
            MODEL_EN_ONNX_URL,
            MODEL_EN_JSON_URL,
        )?;
    }

    match cli {
        Cli::Done => speak("任务完成")?,
        Cli::Confirm => speak("等待确认")?,
        Cli::Plan => speak("需要确认方案")?,
        Cli::Say { text } => speak(&text)?,
    }

    Ok(())
}
