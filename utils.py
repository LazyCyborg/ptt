import os
import tempfile
from typing import Optional, List
import streamlit as st
from config import ALLOWED_EXTENSIONS
import torchaudio
import torch
import math

STREAMLIT_MAX_SIZE = 200 * 1024 * 1024  # 200MB Streamlit limit


def split_audio_file(file_path: str) -> List[str]:
    """
    Split an audio file into chunks of maximum 200MB each.

    Args:
        file_path: Path to the audio file

    Returns:
        List of paths to the chunk files
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Calculate file size per second of audio
        file_size = os.path.getsize(file_path)
        audio_duration = waveform.shape[1] / sample_rate
        bytes_per_second = file_size / audio_duration

        # Calculate maximum chunk duration to stay under 200MB
        max_chunk_duration = STREAMLIT_MAX_SIZE / bytes_per_second

        # Round down to nearest minute for cleaner splitting
        max_chunk_duration = math.floor(max_chunk_duration / 60) * 60

        # Calculate number of chunks needed
        num_chunks = math.ceil(audio_duration / max_chunk_duration)

        chunk_files = []
        for i in range(num_chunks):
            start_time = i * max_chunk_duration
            end_time = min((i + 1) * max_chunk_duration, audio_duration)

            # Calculate sample indices
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract chunk
            chunk_waveform = waveform[:, start_sample:end_sample]

            # Save chunk to temporary file
            chunk_path = tempfile.mktemp(suffix=f'_chunk_{i + 1}.wav')
            torchaudio.save(chunk_path, chunk_waveform, sample_rate)
            chunk_files.append(chunk_path)

        return chunk_files

    except Exception as e:
        st.error(f"Error splitting audio file: {str(e)}")
        return []


def validate_file(uploaded_file) -> tuple[bool, str]:
    """Validate an uploaded file."""
    if uploaded_file is None:
        return False, "No file uploaded"

    # Check file extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"

    return True, ""


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file and split if necessary."""
    try:
        # Save initial file
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Check if file needs splitting
        if uploaded_file.size > STREAMLIT_MAX_SIZE:
            st.info("Large file detected. Splitting into manageable chunks...")
            chunk_files = split_audio_file(temp_path)

            # Clean up original temp file
            os.remove(temp_path)

            if not chunk_files:
                st.error("Failed to split audio file")
                return None

            return chunk_files

        return [temp_path]  # Return as list for consistent interface

    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")

def format_time(seconds: float) -> str:
    """Convert seconds to human-readable time format."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def create_download_link(df, filename: str) -> str:
    """Create a download link for a dataframe."""
    csv = df.to_csv(index=False)
    return f'<a href="data:file/csv;base64,{csv}" download="{filename}">Download {filename}</a>'