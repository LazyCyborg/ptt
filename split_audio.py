#!/usr/bin/env python3
import os
import sys
import argparse
import torchaudio
import torch
import soundfile as sf
import math
from pathlib import Path
import logging
from typing import Optional, List, Tuple, Dict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedAudioSplitter:
    def __init__(self, chunk_duration: float = 60.0, overlap_duration: float = 20.0):
        """
        Initialize AudioSplitter with configurable chunk and overlap durations.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Duration of overlap between chunks in seconds
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with proper format handling."""
        try:
            data, sample_rate = sf.read(file_path)
            logger.info(f"Loaded audio file: {file_path}")
            logger.info(f"Sample rate: {sample_rate}")
            return data, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise

    def split_with_overlap(self, input_file: str, output_dir: str) -> List[Dict]:
        """
        Split audio file into overlapping chunks.

        Returns:
            List of dictionaries containing chunk info:
            {
                'path': str,
                'start_time': float,
                'end_time': float,
                'is_first': bool,
                'is_last': bool
            }
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # Load audio
            audio_data, sample_rate = self.load_audio(input_file)
            duration = len(audio_data) / sample_rate

            # Calculate parameters
            chunk_samples = int(self.chunk_duration * sample_rate)
            overlap_samples = int(self.overlap_duration * sample_rate)
            step_samples = chunk_samples - overlap_samples

            chunks = []
            position = 0
            chunk_number = 1
            base_name = Path(input_file).stem

            while position < len(audio_data):
                # Calculate chunk boundaries
                end_pos = min(position + chunk_samples, len(audio_data))
                chunk_data = audio_data[position:end_pos]

                # Save chunk
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_chunk_{chunk_number:03d}.wav"
                )
                sf.write(output_file, chunk_data, sample_rate)

                # Store chunk info
                chunk_info = {
                    'path': output_file,
                    'start_time': position / sample_rate,
                    'end_time': end_pos / sample_rate,
                    'is_first': chunk_number == 1,
                    'is_last': end_pos == len(audio_data)
                }
                chunks.append(chunk_info)

                # Update position and chunk number
                position += step_samples
                chunk_number += 1

                logger.info(
                    f"Saved chunk {chunk_number - 1}: {output_file} "
                    f"({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)"
                )

            return chunks

        except Exception as e:
            logger.error(f"Error splitting file: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Split audio files into overlapping chunks'
    )
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument(
        '--output-dir', '-o',
        default='split_files',
        help='Output directory for chunks'
    )
    parser.add_argument(
        '--chunk-duration', '-c',
        type=float,
        default=60.0,
        help='Chunk duration in seconds'
    )
    parser.add_argument(
        '--overlap-duration', '-v',
        type=float,
        default=20.0,
        help='Overlap duration in seconds'
    )

    args = parser.parse_args()

    try:
        splitter = AdvancedAudioSplitter(
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap_duration
        )
        chunks = splitter.split_with_overlap(args.input_file, args.output_dir)
        logger.info(f"Successfully split into {len(chunks)} chunks")
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())