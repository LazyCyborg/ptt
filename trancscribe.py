import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from transformers import pipeline
from textacy.preprocessing import normalize, remove, replace, pipeline as pp
from functools import partial
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

@dataclass
class AudioConfig:
    """Configuration for audio processing settings"""
    chunk_duration: float = 30.0  # seconds
    target_rate: int = 16000
    force_resample: bool = False
    lowpass_freq: float = 8000
    highpass_freq: float = 80
    language: str = 'sv'
    min_silence_duration: float = 0.5  # seconds
    silence_threshold: float = 0.1
    enable_smart_chunking: bool = True

@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing settings"""
    normalize_chars: bool = True
    remove_punctuation: bool = False
    max_repeating_chars: Optional[int] = 3
    min_sentence_length: int = 3
    enable_translation: bool = True

@dataclass
class TranscriptionConfig:
    """Configuration for transcription model and processing"""
    model_name: str = "openai/whisper-large-v3"
    use_whisper: bool = True
    device: Optional[str] = None
    batch_size: int = 1
    language: str = 'sv'
    task: str = "transcribe"
    return_timestamps: bool = False


class AudioTextProcessor:
    """
    Enhanced class for preprocessing audio files and transcribing them using ASR models.
    Includes smart chunking and improved error handling.
    """

    def __init__(
            self,
            audio_config: Optional[AudioConfig] = None,
            preprocess_config: Optional[PreprocessConfig] = None,
            transcription_config: Optional[TranscriptionConfig] = None,
            progress_callback: Optional[callable] = None
    ):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store configurations with defaults if not provided
        self.audio_config = audio_config or AudioConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.progress_callback = progress_callback

        # Initialize device and models
        self.device = self._get_device()
        self._initialize_models()
        self.text_preprocessor = self._initialize_text_preprocessor()

        # Initialize results storage
        self.transcriptions = {}
        self.processed_data = pd.DataFrame()

    def _get_device(self) -> Union[str, int]:
        """Determine the appropriate device to use."""
        if self.transcription_config.device is not None:
            return self.transcription_config.device

        if torch.backends.mps.is_available():
            self.logger.info("Using MPS (Apple Silicon) device")
            return 'mps'
        elif torch.cuda.is_available():
            self.logger.info("Using CUDA device")
            return 0
        else:
            self.logger.info("Using CPU device")
            return -1

    def _initialize_models(self) -> None:
        """Initialize transcription and translation models."""
        try:
            # Initialize transcription model
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model=self.transcription_config.model_name,
                device=self.device
            )
            self.logger.info(f"Transcription model loaded: {self.transcription_config.model_name}")

            # Initialize translation model if needed
            if self.preprocess_config.enable_translation and self.audio_config.language != 'en':
                translation_model = f"Helsinki-NLP/opus-mt-{self.audio_config.language}-en"
                self.translator = pipeline(
                    "translation",
                    model=translation_model,
                    device=self.device
                )
                self.logger.info(f"Translation model loaded: {translation_model}")
            else:
                self.translator = None

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file and validate its format.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            speech_tensor, sampling_rate = torchaudio.load(file_path)
            self.logger.info(f"Loaded audio file: {file_path}")
            self.logger.debug(f"Audio shape: {speech_tensor.shape}, Sample rate: {sampling_rate}")
            return speech_tensor, sampling_rate
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {str(e)}")
            raise

    def process_audio(self, speech_tensor: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Process audio with filtering and resampling.

        Args:
            speech_tensor: Input audio tensor
            sampling_rate: Original sampling rate

        Returns:
            Processed audio tensor
        """
        try:
            # Resample if needed
            if self.audio_config.force_resample or sampling_rate != self.audio_config.target_rate:
                resampler = T.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.audio_config.target_rate
                )
                speech_tensor = resampler(speech_tensor)
                self.logger.debug(f"Resampled audio to {self.audio_config.target_rate}Hz")

            # Apply filters
            speech_tensor = F.lowpass_biquad(
                speech_tensor,
                self.audio_config.target_rate,
                self.audio_config.lowpass_freq
            )
            speech_tensor = F.highpass_biquad(
                speech_tensor,
                self.audio_config.target_rate,
                self.audio_config.highpass_freq
            )

            # Convert to mono if needed
            if speech_tensor.size(0) > 1:
                speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)
                self.logger.debug("Converted audio to mono")

            return speech_tensor

        except Exception as e:
            self.logger.error(f"Audio processing failed: {str(e)}")
            raise

    def detect_silence(self, audio: torch.Tensor) -> List[int]:
        """
        Detect silence points in audio for smart chunking.

        Args:
            audio: Input audio tensor

        Returns:
            List of sample indices where silence is detected
        """
        try:
            # Calculate window size for silence detection
            window_size = int(self.audio_config.min_silence_duration *
                              self.audio_config.target_rate)

            # Calculate energy in windows
            audio_flat = audio.view(-1)
            num_windows = audio_flat.size(0) // window_size
            windows = audio_flat[:num_windows * window_size].view(-1, window_size)
            energy = torch.norm(windows, dim=1)

            # Find silence points
            silence_points = torch.where(energy < self.audio_config.silence_threshold)[0]
            silence_points = silence_points * window_size

            return silence_points.tolist()

        except Exception as e:
            self.logger.error(f"Silence detection failed: {str(e)}")
            return []  # Return empty list if silence detection fails

    def _initialize_text_preprocessor(self) -> callable:
        """Initialize the text preprocessing pipeline based on config."""
        steps = [
            normalize.unicode,
            normalize.whitespace,
            normalize.bullet_points,
            remove.html_tags,
            remove.brackets,
            partial(replace.urls, repl="_URL_"),
            partial(replace.emails, repl="_EMAIL_"),
            partial(replace.phone_numbers, repl="_PHONE_"),
            partial(replace.user_handles, repl="_USER_"),
            partial(replace.hashtags, repl="_HASHTAG_"),
            partial(replace.emojis, repl="_EMOJI_"),
            partial(replace.numbers, repl="_NUMBER_"),
            partial(replace.currency_symbols, repl="_CURRENCY_"),
        ]

        if self.preprocess_config.normalize_chars:
            steps.extend([
                normalize.quotation_marks,
                normalize.hyphenated_words,
            ])

        if self.preprocess_config.remove_punctuation:
            steps.append(remove.punctuation)

        if self.preprocess_config.max_repeating_chars:
            steps.append(
                partial(normalize.repeating_chars,
                        chars=self.preprocess_config.max_repeating_chars)
            )

        return pp.make_pipeline(*steps)

    def smart_chunk_audio(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Split audio into chunks intelligently, trying to split at silence points.

        Args:
            audio: Input audio tensor

        Returns:
            List of audio chunk tensors
        """
        try:
            chunk_length = int(self.audio_config.chunk_duration *
                               self.audio_config.target_rate)

            if not self.audio_config.enable_smart_chunking:
                # Simple chunking if smart chunking is disabled
                chunks = []
                audio_flat = audio.view(-1)
                for i in range(0, audio_flat.size(0), chunk_length):
                    chunk = audio_flat[i:min(i + chunk_length, audio_flat.size(0))]
                    if chunk.size(0) < chunk_length:
                        # Pad last chunk if needed
                        chunk = torch.nn.functional.pad(
                            chunk,
                            (0, chunk_length - chunk.size(0))
                        )
                    chunks.append(chunk)
                return chunks

            # Smart chunking
            silence_points = self.detect_silence(audio)
            chunks = []
            start = 0
            audio_flat = audio.view(-1)

            while start < audio_flat.size(0):
                # Find the best silence point near the desired chunk length
                end = start + chunk_length
                if end >= audio_flat.size(0):
                    chunks.append(audio_flat[start:])
                    break

                # Look for silence points in a window around the desired end point
                window = 2 * self.audio_config.target_rate  # 2 second window
                potential_points = [p for p in silence_points
                                    if end - window <= p <= end + window]

                if potential_points:
                    # Use the silence point closest to the desired chunk length
                    split_point = min(potential_points, key=lambda x: abs(x - end))
                else:
                    # If no silence point found, use the exact chunk length
                    split_point = end

                chunks.append(audio_flat[start:split_point])
                start = split_point

            return chunks

        except Exception as e:
            self.logger.error(f"Audio chunking failed: {str(e)}")
            raise

    def transcribe_chunk(self, chunk: torch.Tensor) -> Dict[str, str]:
        """
        Transcribe a single chunk of audio.

        Args:
            chunk: Audio chunk tensor

        Returns:
            Dictionary with original and translated text
        """
        try:
            # Prepare chunk for transcription
            chunk_np = chunk.numpy()

            # Transcribe
            if self.transcription_config.use_whisper:
                result = self.transcriber(
                    chunk_np,
                    generate_kwargs={
                        "task": self.transcription_config.task,
                        "language": self.audio_config.language,
                        "return_timestamps": self.transcription_config.return_timestamps
                    }
                )

                # Extract text and ensure it's a string
                transcribed_text = str(result['text']).strip() if 'text' in result else ""
            else:
                result = self.transcriber(chunk_np)
                transcribed_text = str(result['text']).strip() if 'text' in result else ""

            # Translate if enabled
            translated_text = None
            if (self.preprocess_config.enable_translation and
                    self.translator and
                    transcribed_text):
                try:
                    translation = self.translator(transcribed_text)
                    translated_text = str(translation[0]['translation_text']).strip()
                except Exception as e:
                    self.logger.warning(f"Translation failed: {str(e)}")

            return {
                'original': transcribed_text,
                'translated': translated_text
            }

        except Exception as e:
            self.logger.error(f"Chunk transcription failed: {str(e)}")
            return {'original': '', 'translated': None}

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess transcribed text and split into sentences.

        Args:
            text: Input text to process

        Returns:
            List of processed sentences
        """
        try:
            # Ensure text is string
            text = str(text)

            # Apply preprocessing pipeline
            processed_text = self.text_preprocessor(text)

            # Split into sentences and filter
            sentences = [
                s.strip() for s in processed_text.split('.')
                if len(s.strip().split()) >= self.preprocess_config.min_sentence_length
            ]

            return sentences

        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {str(e)}")
            return [text]  # Return original text if processing fails

    def process_file(self, file_path: str) -> Optional[Dict]:
        """
        Process a single audio file and return results.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary containing processed results or None if processing fails
        """
        try:
            self.logger.info(f"Starting processing of file: {file_path}")

            # Load and process audio
            speech_tensor, sampling_rate = self.load_audio(file_path)
            processed_audio = self.process_audio(speech_tensor, sampling_rate)

            # Get chunks
            chunks = self.smart_chunk_audio(processed_audio)
            total_chunks = len(chunks)

            # Process chunks
            chunk_results = []
            for i, chunk in enumerate(chunks):
                # Update progress if callback provided
                if self.progress_callback:
                    progress = (i + 1) / total_chunks
                    self.progress_callback(progress)

                # Transcribe chunk
                result = self.transcribe_chunk(chunk)
                chunk_results.append(result)

                self.logger.debug(f"Processed chunk {i + 1}/{total_chunks}")

            # Combine results
            original_text = ' '.join(r['original'] for r in chunk_results if r['original'])
            translated_text = ' '.join(r['translated'] for r in chunk_results
                                       if r['translated'] is not None)

            # Process the combined text
            processed_sentences = self.preprocess_text(original_text)

            if processed_sentences:
                result = {
                    'filename': os.path.basename(file_path),
                    'original_text': ' '.join(processed_sentences),
                    'translated_text': translated_text if translated_text else None,
                    'num_sentences': len(processed_sentences),
                    'language': self.audio_config.language,
                    'duration': len(processed_audio.view(-1)) / self.audio_config.target_rate
                }

                self.logger.info(f"Successfully processed file: {file_path}")
                return result

            return None

        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {str(e)}")
            return None

    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Process all audio files in a directory.

        Args:
            directory_path: Path to directory containing audio files

        Returns:
            DataFrame containing results for all processed files
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Get list of audio files
        audio_files = [f for f in os.listdir(directory_path)
                       if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))]

        if not audio_files:
            self.logger.warning(f"No audio files found in {directory_path}")
            return pd.DataFrame()

        # Process each file
        results = []
        total_files = len(audio_files)

        for i, filename in enumerate(audio_files):
            file_path = os.path.join(directory_path, filename)

            # Update overall progress if callback provided
            if self.progress_callback:
                overall_progress = i / total_files
                self.progress_callback(overall_progress)

            result = self.process_file(file_path)
            if result:
                results.append(result)

        # Create DataFrame
        self.processed_data = pd.DataFrame(results)
        return self.processed_data

    def save_results(
            self,
            output_path: str,
            format: str = 'csv',
            include_timestamps: bool = False
    ) -> None:
        """
        Save processing results to file.

        Args:
            output_path: Path to save results
            format: Output format ('csv' or 'hdf')
            include_timestamps: Whether to include processing timestamps
        """
        if self.processed_data.empty:
            self.logger.warning("No data to save")
            return

        try:
            # Add timestamp to filename if requested
            if include_timestamps:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_path = output_path.rsplit('.', 1)[0]
                output_path = f"{base_path}_{timestamp}.{format}"

            # Save in requested format
            if format == 'csv':
                self.processed_data.to_csv(output_path, index=False)
            elif format == 'hdf':
                self.processed_data.to_hdf(
                    output_path,
                    key='transcription_data',
                    mode='w'
                )
            else:
                raise ValueError(f"Unsupported output format: {format}")

            self.logger.info(f"Results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
