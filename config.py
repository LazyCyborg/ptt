from dataclasses import dataclass
from typing import Optional, Dict, List

# Available language options
SUPPORTED_LANGUAGES = {
    'Swedish': 'sv',
    'English': 'en',
    'Norwegian': 'no',
    'Danish': 'da',
    'Finnish': 'fi',
    'German': 'de',
    'French': 'fr',
    'Spanish': 'es'
}

# Available ASR models
AVAILABLE_MODELS = [
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base"
]

# Default configuration values
DEFAULT_CHUNK_DURATION = 30.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_SILENCE = 0.5
DEFAULT_SILENCE_THRESHOLD = 0.1
DEFAULT_MIN_SENTENCE_LENGTH = 3

# File handling settings
ALLOWED_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac']
#MAX_FILE_SIZE = 2000 * 1024 * 1024

# UI Settings
SIDEBAR_WIDTH = 300
MAIN_WIDTH = 800