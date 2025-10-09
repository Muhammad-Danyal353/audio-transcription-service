import argparse
import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/v1/transcribe"


def main():
    p = argparse.ArgumentParser(description="Minimal transcription CLI")
    p.add_argument("audio", type=str, help="Path to audio file")
    p.add_argument("--language_hint", type=str, default=None)
    p.add_argument("--enable_separation", action="store_true", default=False)
    p.add_argument("--model_size", type=str, default="small")
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--url", type=str, default=API_URL)
    args = p.parse_args()

    cfg = {
        "language_hint": args.language_hint,
        "enable_separation": args.enable_separation,
        "model_size": args.model_size,
        "target_sr": args.target_sr,
    }

    path = Path(args.audio)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/octet-stream")}
        data = {"config": json.dumps({k: v for k, v in cfg.items() if v is not None})}
        r = requests.post(args.url, files=files, data=data, timeout=600)
        r.raise_for_status()
        resp = r.json()
        print(resp.get("text", ""))


if __name__ == "__main__":
    main()