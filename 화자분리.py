"""
YouTube → WhisperX 전사(+화자분리).

torch / torchaudio / torchvision 버전·빌드(CPU vs CUDA)가 어긋나면
`Could not load ... libtorchaudio.pyd` 가 납니다. whisperx 3.7.x 는 torch 2.8 대 권장.

CPU 재설치 예:
  pip install -r requirements-e-torch.txt --index-url https://download.pytorch.org/whl/cpu
"""
import os
import subprocess
import torch
import whisperx
import gc
from whisperx.diarize import DiarizationPipeline

# =========================
# 설정
# =========================
CHANNEL_URL = "https://www.youtube.com/"

DOWNLOAD_DIR = "downloads"
TRANSCRIPT_DIR = "transcripts-new"

MODEL_SIZE = "large-v2"   # 최신 예제 기준
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 16
LANGUAGE = None  # auto

HF_TOKEN =""

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# =========================
# 1. 채널 영상 ID + 제목 (--flat-playlist)
# =========================
def get_video_entries(channel_url: str) -> list[tuple[str, str]]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print",
        "%(id)s\t%(title)s",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            vid, title = line.split("\t", 1)
            rows.append((vid.strip(), title.strip()))
        else:
            rows.append((line, ""))
    return rows

# =========================
# 2. 오디오 다운로드
# =========================
def download_audio(video_id: str) -> str:
    wav_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")
    if os.path.exists(wav_path):
        return wav_path

    url = f"https://www.youtube.com/watch?v={video_id}"
    subprocess.run([
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", os.path.join(DOWNLOAD_DIR, f"{video_id}.%(ext)s"),
        url
    ], check=True)

    return wav_path

# =========================
# 3. WhisperX 최신 파이프라인
# =========================
def transcribe_with_diarization(audio_path: str, video_id: str):
    out_txt = os.path.join(TRANSCRIPT_DIR, f"{video_id}.txt")
    if os.path.exists(out_txt):
        return

    audio = whisperx.load_audio(audio_path)

    # 1️⃣ Transcribe
    model = whisperx.load_model(
        MODEL_SIZE,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE
    )

    result = model.transcribe(audio, batch_size=BATCH_SIZE)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2️⃣ Align
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=DEVICE
    )

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 3️⃣ Diarization
    diarize_model = DiarizationPipeline(
        token=HF_TOKEN,
        device=DEVICE
    )

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 저장
    with open(out_txt, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(
                f"[{seg['start']:.2f} → {seg['end']:.2f}] "
                f"{speaker}: {seg['text']}\n"
            )

# =========================
# 실행
# =========================
if __name__ == "__main__":
    entries = get_video_entries(CHANNEL_URL)

    for vid, title in entries:
        print(f"제목: {title}", flush=True)
        if "AI" not in title:
            print("  → 스킵: 제목에 'AI' 없음", flush=True)
            continue

        audio = download_audio(vid)
        transcribe_with_diarization(audio, vid)

    print("🎉 완료")
