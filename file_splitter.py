import os
import sys
import tempfile
import subprocess
import platform
from pathlib import Path
import streamlit as st
from contextlib import contextmanager


@contextmanager
def processing_state():
    """Context manager to handle processing state."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    st.session_state.processing = True
    try:
        yield
    finally:
        st.session_state.processing = False


def validate_path():
    """Validate the input path without triggering a reload"""
    if st.session_state.input_path_key:
        st.session_state.input_path = st.session_state.input_path_key
        st.session_state.path_valid = os.path.exists(st.session_state.input_path_key)


def validate_output():
    """Validate the output directory without triggering a reload"""
    if st.session_state.output_dir_key:
        st.session_state.output_dir = st.session_state.output_dir_key


class FileSplitter:
    def __init__(self):
        self.max_size = 190 * 1024 * 1024  # 190MB to stay safely under 200MB limit

    @staticmethod
    def create_splitter_script(input_file: str, output_dir: str) -> str:
        """Create the Python script for file splitting."""
        script = f"""
import os
import soundfile as sf
import math
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_large_file(input_file: str, output_dir: str):
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get file info
        info = sf.info(input_file)
        total_frames = info.frames
        sample_rate = info.samplerate

        # Calculate chunks
        max_size = 190 * 1024 * 1024  # 190MB
        file_size = os.path.getsize(input_file)
        bytes_per_frame = file_size / total_frames
        frames_per_chunk = int(max_size / bytes_per_frame)

        # Prepare for splitting
        base_name = Path(input_file).stem
        chunk_count = math.ceil(total_frames / frames_per_chunk)

        # Process chunks
        for chunk_idx in range(chunk_count):
            start_frame = chunk_idx * frames_per_chunk
            end_frame = min(start_frame + frames_per_chunk, total_frames)

            # Read chunk
            data, _ = sf.read(input_file, start=start_frame, frames=end_frame - start_frame)

            # Save chunk
            output_file = os.path.join(output_dir, f"{{base_name}}_part{{chunk_idx + 1:03d}}.wav")
            sf.write(output_file, data, sample_rate)

            # Log progress
            chunk_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Saved chunk {{chunk_idx + 1}}/{{chunk_count}}: {{output_file}} ({{chunk_size_mb:.1f}}MB)")

    except Exception as e:
        print(f"Error: {{str(e)}}")
        raise

if __name__ == "__main__":
    input_file = "{input_file}"
    output_dir = "{output_dir}"
    split_large_file(input_file, output_dir)
"""
        return script


def split_audio_gui():
    """Streamlit GUI for audio file splitting."""
    st.title("Large Audio File Splitter")

    st.write("""
    This tool splits large audio files into chunks smaller than 200MB.
    Use this before uploading to the main transcription app.
    """)

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.input_path = ""
        st.session_state.output_dir = os.path.join(os.path.expanduser("~"), "split_audio")
        st.session_state.path_valid = False
        st.session_state.processing = False

    # File path input with callback
    st.text_input(
        "Audio File Path",
        key="input_path_key",
        value=st.session_state.input_path,
        on_change=validate_path
    )

    # Output directory input with callback
    st.text_input(
        "Output Directory",
        key="output_dir_key",
        value=st.session_state.output_dir,
        on_change=validate_output
    )

    # Only show file info if path is valid
    if hasattr(st.session_state, 'path_valid') and st.session_state.path_valid:
        file_size_mb = os.path.getsize(st.session_state.input_path) / (1024 * 1024)
        st.info(f"Selected file size: {file_size_mb:.1f}MB")

        if file_size_mb <= 200:
            st.warning("This file is already under 200MB and doesn't need splitting.")
        else:
            if not st.session_state.get('processing', False):
                if st.button("Split File", use_container_width=True):
                    with processing_state():
                        try:
                            # Create the splitter script
                            splitter_script = FileSplitter.create_splitter_script(
                                st.session_state.input_path,
                                st.session_state.output_dir
                            )

                            # Save the script to a temporary file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                                script_fname = tmp.name
                                tmp.write(splitter_script)

                            # Launch the splitter in a separate process
                            st.info("Starting file splitting process...")
                            process = subprocess.Popen(
                                [sys.executable, script_fname],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )

                            # Show progress
                            with st.spinner("Splitting file..."):
                                output_placeholder = st.empty()
                                all_output = []
                                while True:
                                    output = process.stdout.readline()
                                    if output:
                                        all_output.append(output.strip())
                                        output_placeholder.text("\n".join(all_output))
                                    elif process.poll() is not None:
                                        break

                            # Check if process completed successfully
                            if process.returncode == 0:
                                st.success("File splitting completed successfully!")
                                files = [f for f in os.listdir(st.session_state.output_dir)
                                         if f.startswith(Path(st.session_state.input_path).stem)]
                                if files:
                                    st.subheader("Split Files:")
                                    for file in sorted(files):
                                        file_path = os.path.join(st.session_state.output_dir, file)
                                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                        st.text(f"{file} ({size_mb:.1f}MB)")
                            else:
                                error = process.stderr.read()
                                st.error(f"Error during file splitting: {error}")

                            # Cleanup
                            try:
                                os.remove(script_fname)
                            except:
                                pass

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.info("Processing in progress...")

    elif st.session_state.input_path:
        st.error("Selected file does not exist")


if __name__ == "__main__":
    split_audio_gui()