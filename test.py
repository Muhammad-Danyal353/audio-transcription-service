import os

structure = {
    "audio-transcription-service": {
        "src": {
            "api": [
                "__init__.py",
                "main.py",
                "routes.py",
                "models.py",
                "middleware.py"
            ],
            "core": [
                "__init__.py",
                "config.py",
                "logging.py",
                "exceptions.py"
            ],
            "services": [
                "__init__.py",
                "audio_processor.py",
                "separator.py",
                "transcriber.py",
                "vad.py",
                "diarization.py"
            ],
            "utils": [
                "__init__.py",
                "audio_utils.py",
                "device_utils.py",
                "timing.py"
            ],
            "cli": [
                "__init__.py",
                "client.py"
            ]
        },
        "tests": [
            "__init__.py",
            "conftest.py",
            "test_api.py",
            "test_separator.py",
            "test_transcriber.py",
            "test_integration.py"
        ],
        "sample_audio": [
            "clean_speech.wav",
            "noisy_speech.wav",
            "music_with_vocals.mp3",
            "multi_speaker.wav",
            "README.md"
        ],
        "docker": [
            "Dockerfile",
            "docker-compose.yml"
        ],
        "docs": [
            "ADR.md",
            "API.md",
            "openapi.yaml"
        ],
        ".env.example": None,
        ".gitignore": None,
        ".dockerignore": None,
        "requirements.txt": None,
        "requirements-dev.txt": None,
        "pyproject.toml": None,
        "README.md": None,
        "PROGRESS.md": None
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            os.makedirs(path, exist_ok=True)
            for file in content:
                open(os.path.join(path, file), "a").close()
        elif content is None:
            open(path, "a").close()

if __name__ == "__main__":
    create_structure(".", structure)
    print("✅ Project structure created successfully.")
