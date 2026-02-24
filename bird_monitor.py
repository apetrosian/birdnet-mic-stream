#!/usr/bin/env python3
"""Continuous bird monitoring from microphone using BirdNET deep learning model."""

import argparse
import io
import logging
import queue
import sys
import threading
from datetime import datetime

import sounddevice as sd
import soundfile as sf

try:
    from birdnetlib import RecordingFileObject
    from birdnetlib.analyzer import Analyzer
except ImportError:
    logging.error("Error: birdnetlib not installed.")
    logging.error("Install with: pip install birdnetlib")
    sys.exit(1)


class BirdDetector:
    """Continuous real-time bird detection from microphone."""

    def __init__(self, sample_rate=48000, confidence_threshold=0.5, chunk_duration=10):
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.chunk_duration = chunk_duration
        self.analyzer = Analyzer()

    def record_chunk(self):
        """Record audio chunk from microphone."""
        try:
            audio_data = sd.rec(
                int(self.sample_rate * self.chunk_duration),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocking=True,
            )

            # Flatten if needed
            # if len(audio_data.shape) > 1:
            #     audio_data = audio_data.flatten()

            return audio_data
        except Exception as e:
            logging.error(f"✗ Error recording audio: {e}")
            return None

    def detect(self, audio_data):
        """Detect birds in audio data."""
        if audio_data is None or len(audio_data) == 0:
            return []

        try:
            # Create in-memory WAV buffer
            wav_buffer = io.BytesIO()
            sf.write(
                wav_buffer,
                audio_data,
                self.sample_rate,
                format="WAV",
            )
            wav_buffer.seek(0)

            recording = RecordingFileObject(
                self.analyzer, wav_buffer, min_conf=self.confidence_threshold
            )

            # Analyze recording
            recording.analyze()

            # Return detections
            return recording.detections if recording.detections else []

        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return []


class BirdMonitor:
    """Continuous bird monitoring with parallel recording and detection."""

    def __init__(self, sample_rate=48000, chunk_duration=10, confidence_threshold=0.5):
        """
        Initialize continuous bird monitor.

        Args:
            sample_rate: Audio sample rate
            chunk_duration: Duration in seconds for each detection chunk
            confidence_threshold: Minimum confidence for detection
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.confidence_threshold = confidence_threshold

        self.detector = BirdDetector(
            sample_rate=sample_rate,
            confidence_threshold=confidence_threshold,
            chunk_duration=chunk_duration,
        )

        # Queues for parallel processing
        self.recording_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue()
        self.is_running = False
        self.recording_thread = None
        self.detection_thread = None

    def _recording_thread(self):
        """Thread for continuous audio capture."""
        while self.is_running:
            try:
                audio_data = self.detector.record_chunk()

                if audio_data is not None:
                    self.recording_queue.put(audio_data)
            except Exception as e:
                logging.error(f"Error in recording thread: {e}")
                if self.is_running:
                    continue

    def _detection_thread(self):
        """Thread for detection."""
        while self.is_running:
            try:
                audio_data = self.recording_queue.get(timeout=1)

                if audio_data is not None:
                    detections = self.detector.detect(audio_data)

                    if detections:
                        self.detection_queue.put(
                            {"timestamp": datetime.now(), "detections": detections}
                        )
            except queue.Empty:
                pass
            except Exception as e:
                logging.error(f"Error in detection thread: {e}")
                if self.is_running:
                    continue

    def start(self):
        """Start continuous bird monitoring with parallel recording and detection."""
        logging.info("Starting continuous bird detection")
        logging.info(f"Sample rate: {self.sample_rate}")
        logging.info(f"Chunk duration: {self.chunk_duration}s")
        logging.info(f"Confidence threshold: {self.confidence_threshold}")
        logging.info("Press Ctrl+C to stop\n")

        self.is_running = True

        self.recording_thread = threading.Thread(
            target=self._recording_thread, daemon=False, name="RecordingThread"
        )
        self.detection_thread = threading.Thread(
            target=self._detection_thread, daemon=False, name="DetectionThread"
        )

        self.recording_thread.start()
        self.detection_thread.start()

        try:
            while self.is_running:
                try:
                    result = self.detection_queue.get(timeout=1)
                    self._print_detections(result)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            logging.info("Stopping bird detection...")
            self.stop()

    def stop(self):
        """Stop bird monitoring with proper cleanup."""

        self.is_running = False

        # Drain queues before shutdown
        try:
            while not self.recording_queue.empty():
                self.recording_queue.get_nowait()
            while not self.detection_queue.empty():
                self.detection_queue.get_nowait()
            logging.info("Queues drained")
        except Exception as e:
            logging.error(f"Error draining queues: {e}")

        # Wait for threads to finish with timeout
        threads = [
            self.recording_thread,
            self.detection_thread,
        ]
        for thread in threads:
            if thread is not None:
                thread.join()
                if thread.is_alive():
                    logging.warning(f"Thread {thread.name} did not stop gracefully")

        # Release audio hardware
        try:
            sd.wait()
            logging.info("Audio hardware released")
        except Exception as e:
            logging.error(f"Error releasing audio harware: {e}")

        # Close analyzer and release resources
        try:
            if hasattr(self.detector, "analyzer") and self.detector.analyzer:
                analyzer = self.detector.analyzer
                if hasattr(analyzer, "interpreter") and analyzer.interpreter:
                    try:
                        analyzer.interpreter.reset_all_variables()
                        logging.info("Analyzer resources released")
                    except Exception as e:
                        logging.warning(f"Could not reset analyzer interpreter: {e}")
                else:
                    logging.info("Analyzer resources released")
        except Exception as e:
            logging.error(f"Error releasing analyzer resources: {e}")

        logging.info("Bird detection stopped")

    def _print_detections(self, result):
        """Print detection results."""
        timestamp = result["timestamp"].strftime("%H:%M:%S")
        detections = result["detections"]

        for det in detections:
            confidence_pct = det["confidence"] * 100
            logging.info(
                f"  {timestamp}: {det['common_name']:35} ({confidence_pct:5.1f}%)"
            )


def main():
    """Parse arguments and start monitoring."""

    parser = argparse.ArgumentParser(
        description="Continuous bird detection using BirdNET and microphone"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration in seconds for each detection chunk (default: 10)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Audio sample rate (default: 48000)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        logging.error("Confidence must be between 0.0 and 1.0")
        sys.exit(1)

    if args.duration <= 0:
        logging.error("Duration must be positive")
        sys.exit(1)

    monitor = BirdMonitor(
        chunk_duration=args.duration,
        confidence_threshold=args.confidence,
        sample_rate=args.sample_rate,
    )

    monitor.start()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    main()
