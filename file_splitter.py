#!/usr/bin/env python3
import os
import soundfile as sf
import math
from pathlib import Path
import logging
from typing import List
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileSplitter:
    def __init__(self):
        self.max_size = 190 * 1024 * 1024  # 190MB to stay safely under 200MB limit

    def split_large_file(self, input_file: str, output_dir: str) -> List[str]:
        """
        Split large audio file into chunks smaller than 200MB.

        Args:
            input_file: Path to input audio file
            output_dir: Directory to save split files

        Returns:
            List of paths to split files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get file info
            info = sf.info(input_file)
            total_frames = info.frames
            sample_rate = info.samplerate

            # Calculate frames per chunk based on file size
            file_size = os.path.getsize(input_file)
            bytes_per_frame = file_size / total_frames
            frames_per_chunk = int(self.max_size / bytes_per_frame)

            # Prepare for splitting
            output_files = []
            base_name = Path(input_file).stem

            # Process chunks
            chunk_count = math.ceil(total_frames / frames_per_chunk)

            for chunk_idx in range(chunk_count):
                # Calculate frame range for this chunk
                start_frame = chunk_idx * frames_per_chunk
                end_frame = min(start_frame + frames_per_chunk, total_frames)

                # Read chunk
                data, _ = sf.read(input_file,
                                  start=start_frame,
                                  frames=end_frame - start_frame)

                # Generate output filename
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_part{chunk_idx + 1:03d}.wav"
                )

                # Save chunk
                sf.write(output_file, data, sample_rate)

                chunk_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                logger.info(f"Saved chunk {chunk_idx + 1}/{chunk_count}: "
                            f"{output_file} ({chunk_size_mb:.1f}MB)")

                output_files.append(output_file)

            return output_files

        except Exception as e:
            logger.error(f"Error splitting file: {e}")
            raise


def split_audio_gui():
    """Streamlit GUI for audio file splitting."""
    st.title("Large Audio File Splitter")

    st.write("""
    This tool splits large audio files into chunks smaller than 200MB.
    Use this before uploading to the main transcription app.
    """)

    # File selection
    input_path = st.text_input(
        "Audio File Path",
        help="Enter the full path to your audio file"
    )

    # Output directory selection
    output_dir = st.text_input(
        "Output Directory",
        value=os.path.join(os.path.expanduser("~"), "split_audio"),
        help="Directory where split files will be saved"
    )

    if input_path and os.path.exists(input_path):
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        st.info(f"Selected file size: {file_size_mb:.1f}MB")

        if file_size_mb <= 200:
            st.warning("This file is already under 200MB and doesn't need splitting.")
        else:
            if st.button("Split File"):
                try:
                    with st.spinner("Splitting file..."):
                        splitter = FileSplitter()
                        output_files = splitter.split_large_file(input_path, output_dir)

                        st.success(f"File successfully split into {len(output_files)} parts")

                        # Display results
                        st.subheader("Split Files:")
                        for file in output_files:
                            size_mb = os.path.getsize(file) / (1024 * 1024)
                            st.text(f"{os.path.basename(file)} ({size_mb:.1f}MB)")

                        st.info(
                            "You can now upload these files individually to the "
                            "main transcription app."
                        )

                except Exception as e:
                    st.error(f"Error splitting file: {str(e)}")

    elif input_path:
        st.error("Selected file does not exist")


if __name__ == "__main__":
    split_audio_gui()