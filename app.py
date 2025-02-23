import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import logging
logging.getLogger('torch.distributed.nn.jit.instantiator').setLevel(logging.ERROR)


import streamlit as st
import pandas as pd
import os
import time
from pathlib import Path
import tempfile
from typing import List, Dict, Optional
from split_audio import AdvancedAudioSplitter
from file_splitter import FileSplitter
from trancscribe import (
    AudioTextProcessor,
    AudioConfig,
    PreprocessConfig,
    TranscriptionConfig
)
from config import (
    SUPPORTED_LANGUAGES,
    AVAILABLE_MODELS,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MIN_SILENCE,
    DEFAULT_SILENCE_THRESHOLD,
    DEFAULT_MIN_SENTENCE_LENGTH
)
from utils import format_time


def create_config(settings: dict) -> dict:
    """Create unified configuration from settings."""
    return {
        'audio': {
            'chunk_duration': settings['chunk_duration'],
            'target_rate': settings['sample_rate'],
            'language': SUPPORTED_LANGUAGES[settings['language']],
            'min_silence_duration': settings.get('min_silence'),
            'silence_threshold': settings.get('silence_threshold'),
            'enable_smart_chunking': settings['enable_smart_chunking']
        },
        'preprocess': {
            'enable_translation': settings['enable_translation'],
            'min_sentence_length': DEFAULT_MIN_SENTENCE_LENGTH
        },
        'transcription': {
            'model_name': settings['model'],
            'use_whisper': True,
            'return_timestamps': True
        }
    }


def initialize_processor(config: dict) -> Optional[AudioTextProcessor]:
    """Initialize the audio processor with given configuration."""
    try:
        audio_config = AudioConfig(
            chunk_duration=config['audio']['chunk_duration'],
            target_rate=config['audio']['target_rate'],
            language=config['audio']['language'],
            min_silence_duration=config['audio'].get('min_silence_duration'),
            silence_threshold=config['audio'].get('silence_threshold'),
            enable_smart_chunking=config['audio']['enable_smart_chunking']
        )

        preprocess_config = PreprocessConfig(
            enable_translation=config['preprocess']['enable_translation'],
            min_sentence_length=config['preprocess']['min_sentence_length']
        )

        transcription_config = TranscriptionConfig(
            model_name=config['transcription']['model_name'],
            use_whisper=config['transcription']['use_whisper'],
            return_timestamps=config['transcription']['return_timestamps']
        )

        return AudioTextProcessor(
            audio_config=audio_config,
            preprocess_config=preprocess_config,
            transcription_config=transcription_config
        )
    except Exception as e:
        st.error(f"Error initializing processor: {str(e)}")
        return None


def process_overlapping_chunks(
        chunks: List[Dict],
        processor: AudioTextProcessor,
        enable_translation: bool
) -> Optional[Dict]:
    """Process overlapping chunks and combine results intelligently."""
    try:
        all_original_texts = []
        all_translated_texts = []

        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            with st.spinner(f"Processing chunk {i + 1}/{len(chunks)}..."):
                result = processor.process_file(chunk['path'])

                if result:
                    # Handle original text
                    text = result['original_text']
                    if not chunk['is_first'] and len(all_original_texts) > 0:
                        # Remove overlapping portion
                        words = text.split()
                        text = ' '.join(words[len(words) // 3:])
                    all_original_texts.append(text)

                    # Handle translation if enabled
                    if enable_translation and 'translated_text' in result:
                        trans_text = result['translated_text']
                        if not chunk['is_first'] and len(all_translated_texts) > 0:
                            words = trans_text.split()
                            trans_text = ' '.join(words[len(words) // 3:])
                        all_translated_texts.append(trans_text)

                progress_bar.progress((i + 1) / len(chunks))

        combined_result = {
            'original_text': ' '.join(all_original_texts),
            'translated_text': ' '.join(all_translated_texts) if enable_translation else None,
            'duration': chunks[-1]['end_time']
        }

        return combined_result

    except Exception as e:
        st.error(f"Error processing chunks: {str(e)}")
        return None


def display_results(results: Dict):
    """Display processing results."""
    st.header("Transcription Results")

    # Display original text
    st.subheader("Original Text")
    st.write(results['original_text'])

    # Display translation if available
    if results.get('translated_text'):
        st.subheader("English Translation")
        st.write(results['translated_text'])

    # Display duration
    st.info(f"Audio Duration: {format_time(results['duration'])}")

    # Create downloadable files
    st.subheader("Download Options")

    # Text file download
    text_content = results['original_text']
    if results.get('translated_text'):
        text_content += "\n\n=== English Translation ===\n\n" + results['translated_text']

    st.download_button(
        "Download Text",
        text_content,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # CSV download
    df = pd.DataFrame([results])
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        file_name="transcription_results.csv",
        mime="text/csv"
    )


def main():
    st.title("Audio Transcription Tool 🎙️")

    # Instructions
    with st.expander("📖 Instructions", expanded=True):
        st.markdown("""
        ### Instructions:
        1. Select your audio file
        2. Configure your settings in the sidebar
        3. Click 'Process Audio'

        The file will be automatically split into 60-second chunks with smart overlap
        handling for optimal transcription accuracy.

        ### Features:
        - Smart chunking with overlap handling
        - Multiple language support
        - Optional English translation
        - Advanced audio processing options
        - Downloadable results in multiple formats
        """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        settings = {}

        # Basic settings
        settings['language'] = st.selectbox(
            "Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )

        settings['model'] = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0
        )

        settings['enable_translation'] = st.checkbox(
            "Translate to English",
            value=True
        )

        st.sidebar.markdown("---")
        if 'show_splitter' not in st.session_state:
            st.session_state.show_splitter = False

        if st.sidebar.button("🔪 Split Large Files"):
            st.session_state.show_splitter = not st.session_state.show_splitter

        if st.session_state.show_splitter:
            from file_splitter import split_audio_gui
            split_audio_gui()
            if st.sidebar.button("Return to Main"):
                st.session_state.show_splitter = False
                st.rerun()

        # Advanced settings
        with st.expander("Advanced Settings"):
            settings['chunk_duration'] = st.slider(
                "Base Chunk Duration (seconds)",
                min_value=30.0,
                max_value=120.0,
                value=60.0,
                step=10.0
            )

            settings['overlap_duration'] = st.slider(
                "Overlap Duration (seconds)",
                min_value=10.0,
                max_value=30.0,
                value=20.0,
                step=5.0
            )

            settings['sample_rate'] = st.select_slider(
                "Sample Rate",
                options=[8000, 16000, 22050, 44100],
                value=DEFAULT_SAMPLE_RATE
            )

            settings['enable_smart_chunking'] = st.checkbox(
                "Enable Smart Chunking",
                value=True
            )

            if settings['enable_smart_chunking']:
                settings['min_silence'] = st.slider(
                    "Minimum Silence Duration (seconds)",
                    min_value=0.1,
                    max_value=2.0,
                    value=DEFAULT_MIN_SILENCE,
                    step=0.1
                )

                settings['silence_threshold'] = st.slider(
                    "Silence Threshold",
                    min_value=0.01,
                    max_value=0.5,
                    value=DEFAULT_SILENCE_THRESHOLD,
                    step=0.01
                )

    # Main content area
    uploaded_file = st.file_uploader(
        "Select Audio File",
        type=['wav', 'mp3', 'm4a', 'flac']
    )

    if uploaded_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_path = tmp_file.name

        if st.button("Process Audio", type="primary"):
            try:
                start_time = time.time()

                # Create temp directory for chunks
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Split audio
                    with st.spinner("Splitting audio file..."):
                        splitter = AdvancedAudioSplitter(
                            chunk_duration=settings['chunk_duration'],
                            overlap_duration=settings['overlap_duration']
                        )
                        chunks = splitter.split_with_overlap(input_path, temp_dir)
                        st.success(f"Split into {len(chunks)} chunks")

                    # Initialize processor
                    config = create_config(settings)
                    processor = initialize_processor(config)

                    if processor:
                        # Process chunks
                        results = process_overlapping_chunks(
                            chunks,
                            processor,
                            settings['enable_translation']
                        )

                        if results:
                            processing_time = time.time() - start_time
                            st.success(f"Processing completed in {format_time(processing_time)}")

                            # Display results
                            display_results(results)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

            finally:
                # Cleanup
                if os.path.exists(input_path):
                    os.remove(input_path)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Audio Transcription Tool",
        page_icon="🎙️",
        layout="wide"
    )
    main()