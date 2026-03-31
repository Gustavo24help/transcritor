#!/usr/bin/env python3
"""
Pipeline para transcrever mídias exportadas do WhatsApp Business e inserir
as transcrições em arquivos CSV de conversa.

Principais recursos:
- Varre pasta de mídias por .opus/.ogg/.mp4/.mov/.m4a/.aac/.mp3/.wav
- Converte para mp3 usando ffmpeg
- Transcreve com OpenAI em paralelo
- Mantém cache local para retomada
- Gera transcricoes.csv consolidado
- Encaixa [ TRANSCRIÇÃO: ... ] em linhas de áudio nos CSVs de conversa
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

MEDIA_EXTS = {".opus", ".ogg", ".mp4", ".mov", ".m4a", ".aac", ".mp3", ".wav"}
AUDIO_HINT_RE = re.compile(r"\[(?:áudio|audio)\]", re.IGNORECASE)


@dataclass
class TranscriptionRow:
    media_path: str
    media_name: str
    media_hash: str
    converted_mp3: str
    modified_at: str
    transcript: str
    status: str
    error: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "media_path": self.media_path,
            "media_name": self.media_name,
            "media_hash": self.media_hash,
            "converted_mp3": self.converted_mp3,
            "modified_at": self.modified_at,
            "transcript": self.transcript,
            "status": self.status,
            "error": self.error,
        }


class CacheStore:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.lock = threading.Lock()
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                self.data: dict[str, dict[str, str]] = json.load(f)
        else:
            self.data = {}

    def get(self, key: str) -> dict[str, str] | None:
        with self.lock:
            return self.data.get(key)

    def set(self, key: str, value: dict[str, str]) -> None:
        with self.lock:
            self.data[key] = value
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            tmp.replace(self.cache_path)


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg não encontrado no PATH. Instale e valide com: ffmpeg -version"
        )


def sha1_of_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def convert_to_mp3(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha ao converter {src.name}: {proc.stderr.strip()[:400]}")


def transcribe_mp3(client: OpenAI, mp3_path: Path, model: str) -> str:
    with mp3_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="text",
        )
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", "")
    return str(text).strip()


def collect_media_files(media_dir: Path) -> list[Path]:
    files = []
    for p in media_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            files.append(p)
    return sorted(files)


def process_media_file(
    media_path: Path,
    converted_dir: Path,
    client: OpenAI,
    model: str,
    cache: CacheStore,
) -> TranscriptionRow:
    media_hash = sha1_of_file(media_path)
    cached = cache.get(media_hash)
    if cached and cached.get("status") == "ok":
        return TranscriptionRow(**cached)

    safe_name = media_path.stem + "_" + media_hash[:10] + ".mp3"
    mp3_path = converted_dir / safe_name

    try:
        if not mp3_path.exists():
            convert_to_mp3(media_path, mp3_path)
        transcript = transcribe_mp3(client, mp3_path, model)
        row = TranscriptionRow(
            media_path=str(media_path),
            media_name=media_path.name,
            media_hash=media_hash,
            converted_mp3=str(mp3_path),
            modified_at=datetime.fromtimestamp(media_path.stat().st_mtime).isoformat(),
            transcript=transcript,
            status="ok",
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        row = TranscriptionRow(
            media_path=str(media_path),
            media_name=media_path.name,
            media_hash=media_hash,
            converted_mp3=str(mp3_path),
            modified_at=datetime.fromtimestamp(media_path.stat().st_mtime).isoformat(),
            transcript="",
            status="error",
            error=str(exc),
        )

    cache.set(media_hash, row.as_dict())
    return row


def write_transcriptions_csv(rows: list[TranscriptionRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "media_path",
                "media_name",
                "media_hash",
                "converted_mp3",
                "modified_at",
                "transcript",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_dict())


def detect_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def parse_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def should_inject_transcript(msg: str) -> bool:
    msg = (msg or "").strip()
    return bool(AUDIO_HINT_RE.search(msg) or msg.lower() in {"audio", "áudio", ""})


def load_transcription_index(rows: list[TranscriptionRow]) -> dict[str, list[TranscriptionRow]]:
    idx: dict[str, list[TranscriptionRow]] = {}
    for row in rows:
        if row.status != "ok" or not row.transcript:
            continue
        key = row.media_name.lower()
        idx.setdefault(key, []).append(row)
    return idx


def insert_transcriptions_in_csvs(
    csv_input_dir: Path,
    csv_output_dir: Path,
    transcriptions: list[TranscriptionRow],
) -> None:
    idx_by_media_name = load_transcription_index(transcriptions)

    csv_output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(csv_input_dir.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not reader.fieldnames:
                continue
            fieldnames = list(reader.fieldnames)

        message_col = detect_column(fieldnames, ["mensagem", "message", "conteudo", "texto"])
        media_col = detect_column(fieldnames, ["midia", "media", "arquivo", "media_file", "anexo"])

        if not message_col:
            continue

        if "transcricao_audio" not in fieldnames:
            fieldnames.append("transcricao_audio")
        if "mensagem_final" not in fieldnames:
            fieldnames.append("mensagem_final")

        for row in rows:
            msg = row.get(message_col, "")
            row.setdefault("transcricao_audio", "")
            row.setdefault("mensagem_final", msg)

            if not should_inject_transcript(msg):
                continue

            media_value = (row.get(media_col, "") if media_col else "").strip()
            media_name = Path(media_value).name.lower() if media_value else ""

            selected: TranscriptionRow | None = None
            if media_name and media_name in idx_by_media_name:
                selected = idx_by_media_name[media_name].pop(0)
            elif len(idx_by_media_name) == 1:
                # fallback simples quando existe apenas um arquivo de mídia no lote
                only_key = next(iter(idx_by_media_name.keys()))
                if idx_by_media_name[only_key]:
                    selected = idx_by_media_name[only_key].pop(0)

            if not selected:
                continue

            transcript = selected.transcript.strip()
            row["transcricao_audio"] = transcript
            row["mensagem_final"] = f"[ TRANSCRIÇÃO: {transcript} ]"

        out_name = csv_path.stem + "_com_transcricoes.csv"
        out_path = csv_output_dir / out_name
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcrição de mídias WhatsApp + merge em CSVs.")
    p.add_argument("--media-dir", required=True, type=Path, help="Pasta com mídias exportadas")
    p.add_argument("--csv-dir", required=True, type=Path, help="Pasta com CSVs de conversa")
    p.add_argument("--output-dir", required=True, type=Path, help="Pasta de saída")
    p.add_argument("--workers", type=int, default=5, help="Chamadas paralelas à API")
    p.add_argument("--model", default="gpt-4o-mini-transcribe", help="Modelo de transcrição")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key (ou via OPENAI_API_KEY)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Defina OPENAI_API_KEY no ambiente ou passe --api-key.")

    require_ffmpeg()
    client = OpenAI(api_key=args.api_key)

    media_files = collect_media_files(args.media_dir)
    if not media_files:
        raise SystemExit("Nenhuma mídia encontrada na pasta informada.")

    converted_dir = args.output_dir / "convertidos_mp3"
    cache = CacheStore(args.output_dir / "cache_transcricoes.json")

    rows: list[TranscriptionRow] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(process_media_file, media, converted_dir, client, args.model, cache)
            for media in media_files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Transcrevendo"):
            rows.append(fut.result())

    rows.sort(key=lambda r: r.media_path)
    trans_csv = args.output_dir / "transcricoes.csv"
    write_transcriptions_csv(rows, trans_csv)

    insert_transcriptions_in_csvs(
        csv_input_dir=args.csv_dir,
        csv_output_dir=args.output_dir,
        transcriptions=rows,
    )

    ok = sum(1 for r in rows if r.status == "ok")
    err = sum(1 for r in rows if r.status != "ok")
    print(f"Concluído. OK={ok} ERRO={err}")
    print(f"Transcrições: {trans_csv}")


if __name__ == "__main__":
    main()
