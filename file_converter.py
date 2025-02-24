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
    if 'converting' not in st.session_state:
        st.session_state.converting = False
    st.session_state.converting = True
    try:
        yield
    finally:
        st.session_state.converting = False


def validate_convert_path():
    """Validate the input path without triggering a reload"""
    if st.session_state.convert_input_path_key:
        st.session_state.convert_input_path = st.session_state.convert_input_path_key
        st.session_state.convert_path_valid = os.path.exists(st.session_state.convert_input_path_key)


def validate_convert_output():
    """Validate the output directory without triggering a reload"""
    if st.session_state.convert_output_dir_key:
        st.session_state.convert_output_dir = st.session_state.convert_output_dir_key


class AudioConverter:
    @staticmethod
    def create_converter_script(input_file: str, output_dir: str) -> str:
        """Create the Python script for audio conversion."""
        output_file = os.path.join(output_dir, f"{Path(input_file).stem}.wav")
        script = f"""
import os
import subprocess
from pathlib import Path

def convert_audio(input_file, output_file):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(Path(output_file).parent, exist_ok=True)

        # Use ffmpeg to convert the file
        cmd = ['ffmpeg', '-i', input_file, '-y', output_file]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Print the output
        for line in process.stderr:
            print(line.strip())

        # Wait for completion
        process.wait()

        if process.returncode == 0:
            print(f"Successfully converted {{input_file}} to {{output_file}}")
            return True
        else:
            print(f"Error converting file. FFmpeg returned code {{process.returncode}}")
            return False

    except Exception as e:
        print(f"Error: {{str(e)}}")
        return False

if __name__ == "__main__":
    input_file = "{input_file}"
    output_file = "{output_file}"
    convert_audio(input_file, output_file)
"""
        return script, output_file


def convert_audio_gui():
    """Streamlit GUI for audio file conversion."""
    st.title("Audio File Converter")

    st.write("""
    This tool converts audio files to WAV format.
    Use this for converting M4A files before uploading to the main transcription app.
    """)

    # Initialize session state
    if 'convert_initialized' not in st.session_state:
        st.session_state.convert_initialized = True
        st.session_state.convert_input_path = ""
        st.session_state.convert_output_dir = os.path.join(os.path.expanduser("~"), "converted_audio")
        st.session_state.convert_path_valid = False
        st.session_state.converting = False

    # File path input with callback
    st.text_input(
        "Audio File Path",
        key="convert_input_path_key",
        value=st.session_state.convert_input_path,
        on_change=validate_convert_path
    )

    # Output directory input with callback
    st.text_input(
        "Output Directory",
        key="convert_output_dir_key",
        value=st.session_state.convert_output_dir,
        on_change=validate_convert_output
    )

    # Only show file info if path is valid
    if hasattr(st.session_state, 'convert_path_valid') and st.session_state.convert_path_valid:
        file_ext = Path(st.session_state.convert_input_path).suffix.lower()
        st.info(f"Selected file type: {file_ext}")

        if not st.session_state.get('converting', False):
            if st.button("Convert to WAV", use_container_width=True):
                with processing_state():
                    try:
                        # Create the converter script
                        converter_script, output_file = AudioConverter.create_converter_script(
                            st.session_state.convert_input_path,
                            st.session_state.convert_output_dir
                        )

                        # Save the script to a temporary file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                            script_fname = tmp.name
                            tmp.write(converter_script)

                        # Launch the converter in a separate process
                        st.info("Starting audio conversion...")
                        process = subprocess.Popen(
                            [sys.executable, script_fname],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )

                        # Show progress
                        with st.spinner("Converting audio file..."):
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
                            if os.path.exists(output_file):
                                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                                st.success(f"Conversion completed successfully!")
                                st.subheader("Converted File:")
                                st.text(f"{Path(output_file).name} ({size_mb:.1f}MB)")
                            else:
                                st.error("Conversion process completed but output file was not found.")
                        else:
                            error = process.stderr.read()
                            st.error(f"Error during conversion: {error}")

                        # Cleanup
                        try:
                            os.remove(script_fname)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Conversion in progress...")

    elif st.session_state.convert_input_path:
        st.error("Selected file does not exist")


if __name__ == "__main__":
    convert_audio_gui()