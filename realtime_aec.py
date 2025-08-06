# -*- coding: utf-8 -*-
"""
Real-time audio processing script using DTLN-AEC model with microphone input.
This script captures audio from a microphone device and processes it in real-time
using the trained DTLN-AEC model for acoustic echo cancellation.

Requirements:
    - sounddevice
    - soundfile
    - numpy
    - tensorflow-lite
    - A trained DTLN-AEC model (model_1.tflite and model_2.tflite)

Example call:
    python realtime_aec.py -m /path/to/model -d 1 --output_file output.wav

Author: Based on Nils L. Westhausen's DTLN-AEC implementation
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse
import tensorflow.lite as tflite
import threading
import queue
import time
import os

# Make GPUs invisible for TensorFlow Lite
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class RealTimeAEC:
    def __init__(self, model_path=None, sample_rate=16000, block_size=128, input_device=None, output_device=None, load_models=True):
        """
        Initialize the real-time AEC processor.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the model files (without _1.tflite/_2.tflite suffix)
        sample_rate : int
            Sample rate for audio processing (must be 16kHz for DTLN-AEC)
        block_size : int
            Block size for real-time processing
        input_device : int or None
            Audio input device index
        output_device : int or None
            Audio output device index
        load_models : bool
            Whether to load TensorFlow Lite models (set to False for device listing only)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.block_len = 512
        self.block_shift = 128
        self.input_device = input_device
        self.output_device = output_device
        
        # Initialize model-related attributes
        self.interpreter_1 = None
        self.interpreter_2 = None
        self.input_details_1 = None
        self.output_details_1 = None
        self.input_details_2 = None
        self.output_details_2 = None
        self.states_1 = None
        self.states_2 = None
        
        # Load TensorFlow Lite models if requested
        if load_models and model_path:
            self.interpreter_1 = tflite.Interpreter(model_path=model_path + "_1.tflite")
            self.interpreter_1.allocate_tensors()
            self.interpreter_2 = tflite.Interpreter(model_path=model_path + "_2.tflite")
            self.interpreter_2.allocate_tensors()
            
            # Get model details
            self.input_details_1 = self.interpreter_1.get_input_details()
            self.output_details_1 = self.interpreter_1.get_output_details()
            self.input_details_2 = self.interpreter_2.get_input_details()
            self.output_details_2 = self.interpreter_2.get_output_details()
            
            # Initialize LSTM states
            self.states_1 = np.zeros(self.input_details_1[1]["shape"]).astype("float32")
            self.states_2 = np.zeros(self.input_details_2[1]["shape"]).astype("float32")
        
        # Initialize buffers
        self.in_buffer = np.zeros(self.block_len).astype("float32")
        self.in_buffer_lpb = np.zeros(self.block_len).astype("float32")
        self.out_buffer = np.zeros(self.block_len).astype("float32")
        
        # Queues for audio data
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.lpb_queue = queue.Queue()  # Loopback audio queue
        
        # Control flags
        self.is_processing = False
        self.recording = False
        
        # For saving output
        self.output_data = []
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio input status: {status}")
        
        # Add input data to queue
        self.input_queue.put(indata[:, 0].copy())  # Take first channel only
        
        # For this example, we'll use the same input as loopback
        # In a real scenario, you would capture the loopback audio separately
        self.lpb_queue.put(indata[:, 0].copy())
    
    def process_audio_block(self, mic_audio, lpb_audio):
        """
        Process a single audio block through the DTLN-AEC model.
        
        Parameters:
        -----------
        mic_audio : np.ndarray
            Microphone audio block
        lpb_audio : np.ndarray
            Loopback audio block
            
        Returns:
        --------
        np.ndarray
            Processed audio block
        """
        # Update input buffer
        self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
        self.in_buffer[-self.block_shift:] = mic_audio
        
        # Update loopback buffer
        self.in_buffer_lpb[:-self.block_shift] = self.in_buffer_lpb[self.block_shift:]
        self.in_buffer_lpb[-self.block_shift:] = lpb_audio
        
        # Calculate FFT of input block
        in_block_fft = np.fft.rfft(self.in_buffer).astype("complex64")
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
        
        # Calculate FFT of loopback block
        lpb_block_fft = np.fft.rfft(self.in_buffer_lpb).astype("complex64")
        lpb_mag = np.abs(lpb_block_fft)
        lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
        
        # First model inference
        self.interpreter_1.set_tensor(self.input_details_1[0]["index"], in_mag)
        self.interpreter_1.set_tensor(self.input_details_1[2]["index"], lpb_mag)
        self.interpreter_1.set_tensor(self.input_details_1[1]["index"], self.states_1)
        
        self.interpreter_1.invoke()
        
        out_mask = self.interpreter_1.get_tensor(self.output_details_1[0]["index"])
        self.states_1 = self.interpreter_1.get_tensor(self.output_details_1[1]["index"])
        
        # Apply mask and inverse FFT
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
        
        # Prepare loopback input for second model
        in_lpb = np.reshape(self.in_buffer_lpb, (1, 1, -1)).astype("float32")
        
        # Second model inference
        self.interpreter_2.set_tensor(self.input_details_2[1]["index"], self.states_2)
        self.interpreter_2.set_tensor(self.input_details_2[0]["index"], estimated_block)
        self.interpreter_2.set_tensor(self.input_details_2[2]["index"], in_lpb)
        
        self.interpreter_2.invoke()
        
        out_block = self.interpreter_2.get_tensor(self.output_details_2[0]["index"])
        self.states_2 = self.interpreter_2.get_tensor(self.output_details_2[1]["index"])
        
        # Update output buffer
        self.out_buffer[:-self.block_shift] = self.out_buffer[self.block_shift:]
        self.out_buffer[-self.block_shift:] = np.zeros(self.block_shift)
        self.out_buffer += np.squeeze(out_block)
        
        # Extract processed audio
        processed_audio = self.out_buffer[:self.block_shift].copy()
        
        return processed_audio
    
    def processing_thread(self):
        """Main processing thread that handles audio processing."""
        print("Processing thread started...")
        
        while self.is_processing:
            try:
                # Get audio data from queues
                if not self.input_queue.empty() and not self.lpb_queue.empty():
                    mic_data = self.input_queue.get(timeout=0.1)
                    lpb_data = self.lpb_queue.get(timeout=0.1)
                    
                    # Process the audio block
                    processed_audio = self.process_audio_block(mic_data, lpb_data)
                    
                    # Check for clipping and normalize if necessary
                    if np.max(np.abs(processed_audio)) > 0.99:
                        processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.99
                    
                    # Store for saving later
                    if self.recording:
                        self.output_data.append(processed_audio)
                    
                    # Put processed audio in output queue for potential playback
                    self.output_queue.put(processed_audio)
                    
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                break
                
        print("Processing thread stopped.")
    
    def start_processing(self, duration=None, output_file=None, enable_playback=False):
        """
        Start real-time audio processing.
        
        Parameters:
        -----------
        duration : float or None
            Duration in seconds to process (None for indefinite)
        output_file : str or None
            Path to save processed audio
        enable_playback : bool
            Whether to play processed audio through output device
        """
        self.recording = output_file is not None
        self.output_data = []
        
        # Display device information
        print("=" * 60)
        print("REAL-TIME DTLN-AEC AUDIO PROCESSING")
        print("=" * 60)
        
        # Get and display input device info
        input_device_id = self.input_device if self.input_device is not None else sd.default.device[0]
        input_info = self.get_device_info(input_device_id, "input")
        if input_info:
            print(f"INPUT DEVICE:")
            print(f"  ID: {input_info['id']}")
            print(f"  Name: {input_info['name']}")
            print(f"  Channels: {input_info['channels']}")
            print(f"  Default Sample Rate: {input_info['sample_rate']} Hz")
            if self.input_device is None:
                print(f"  Note: Using system default input device")
            else:
                print(f"  Note: Manually selected input device")
        
        # Get and display output device info if playback is enabled
        if enable_playback:
            output_device_id = self.output_device if self.output_device is not None else sd.default.device[1]
            output_info = self.get_device_info(output_device_id, "output")
            if output_info:
                print(f"\nOUTPUT DEVICE:")
                print(f"  ID: {output_info['id']}")
                print(f"  Name: {output_info['name']}")
                print(f"  Channels: {output_info['channels']}")
                print(f"  Default Sample Rate: {output_info['sample_rate']} Hz")
                if self.output_device is None:
                    print(f"  Note: Using system default output device")
                else:
                    print(f"  Note: Manually selected output device")
        
        print(f"\nPROCESSING SETTINGS:")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Block size: {self.block_size}")
        print(f"  Processing block length: {self.block_len}")
        print(f"  Block shift: {self.block_shift}")
        
        if self.recording:
            print(f"  Recording to: {output_file}")
        if enable_playback:
            print(f"  Real-time playback: Enabled")
        
        print("-" * 60)
        
        # Start processing thread
        self.is_processing = True
        processing_thread = threading.Thread(target=self.processing_thread)
        processing_thread.start()
        
        try:
            if enable_playback:
                # Use duplex stream for simultaneous input/output
                print("Starting duplex audio stream (input + output)...")
                with sd.Stream(
                    device=(self.input_device, self.output_device),
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    callback=self.duplex_callback
                ):
                    self._run_processing_loop(duration)
            else:
                # Use input-only stream
                print("Starting input audio stream...")
                with sd.InputStream(
                    device=self.input_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    callback=self.audio_callback
                ):
                    self._run_processing_loop(duration)
                    
        except KeyboardInterrupt:
            print("\nStopping audio processing...")
        except Exception as e:
            print(f"Error during audio processing: {e}")
        finally:
            # Clean up
            self.is_processing = False
            processing_thread.join()
            
            # Save processed audio if requested
            if output_file and self.output_data:
                print(f"Saving processed audio to {output_file}...")
                processed_audio = np.concatenate(self.output_data)
                sf.write(output_file, processed_audio, self.sample_rate)
                print(f"Saved {len(processed_audio)/self.sample_rate:.2f} seconds of processed audio.")
    
    def duplex_callback(self, indata, outdata, frames, time, status):
        """Callback function for duplex audio (input + output)."""
        if status:
            print(f"Audio duplex status: {status}")
        
        # Add input data to queue
        self.input_queue.put(indata[:, 0].copy())  # Take first channel only
        
        # For this example, we'll use the same input as loopback
        # In a real scenario, you would capture the loopback audio separately
        self.lpb_queue.put(indata[:, 0].copy())
        
        # Check if we're in test mode (play input directly)
        if hasattr(self, 'test_mode') and self.test_mode:
            outdata[:, 0] = indata[:, 0] * 0.5  # Play input directly with reduced volume
            return
        
        # Get processed audio for output
        try:
            if not self.output_queue.empty():
                processed_audio = self.output_queue.get_nowait()
                
                # Debug information
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    input_level = np.max(np.abs(indata[:, 0]))
                    output_level = np.max(np.abs(processed_audio))
                    print(f"Input level: {input_level:.4f}, Output level: {output_level:.4f}, Queue size: {self.output_queue.qsize()}")
                
                # Ensure the output matches the expected frame size
                if len(processed_audio) == frames:
                    # Apply some gain to make the output more audible
                    outdata[:, 0] = processed_audio * 2.0  # Increase volume
                else:
                    # Pad or truncate if necessary
                    if len(processed_audio) < frames:
                        padded = np.zeros(frames)
                        padded[:len(processed_audio)] = processed_audio * 2.0  # Increase volume
                        outdata[:, 0] = padded
                    else:
                        outdata[:, 0] = processed_audio[:frames] * 2.0  # Increase volume
            else:
                # No processed audio available, output silence
                outdata[:, 0] = 0
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print("No processed audio available in queue")
        except queue.Empty:
            outdata[:, 0] = 0
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print("Queue empty exception")
    
    def _run_processing_loop(self, duration):
        """Run the main processing loop."""
        print("Audio processing started. Press Ctrl+C to stop...")
        
        if duration:
            time.sleep(duration)
        else:
            # Run until interrupted
            while self.is_processing:
                time.sleep(0.1)
    
    def list_audio_devices(self):
        """List available audio input and output devices."""
        print("Available audio devices:")
        print("=" * 50)
        devices = sd.query_devices()
        
        print("INPUT DEVICES:")
        print("-" * 30)
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  {i}: {device['name']}{default_marker}")
                print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']} Hz")
        
        print("\nOUTPUT DEVICES:")
        print("-" * 30)
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                default_marker = " (DEFAULT)" if i == sd.default.device[1] else ""
                print(f"  {i}: {device['name']}{default_marker}")
                print(f"      Channels: {device['max_output_channels']}, Sample Rate: {device['default_samplerate']} Hz")
        
        print(f"\nCurrent default devices:")
        print(f"  Input: {sd.default.device[0]} - {devices[sd.default.device[0]]['name']}")
        print(f"  Output: {sd.default.device[1]} - {devices[sd.default.device[1]]['name']}")
    
    def get_device_info(self, device_id, device_type="input"):
        """Get information about a specific device."""
        try:
            devices = sd.query_devices()
            if 0 <= device_id < len(devices):
                device = devices[device_id]
                return {
                    'id': device_id,
                    'name': device['name'],
                    'channels': device[f'max_{device_type}_channels'],
                    'sample_rate': device['default_samplerate']
                }
        except Exception as e:
            print(f"Error getting device info: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Real-time DTLN-AEC audio processing")
    parser.add_argument("--model", "-m", required=False, 
                       help="Path to model files (without _1.tflite/_2.tflite suffix)")
    parser.add_argument("--device", "-d", type=int, default=None,
                       help="Input audio device index (use --list_devices to see options)")
    parser.add_argument("--output_device", "-od", type=int, default=None,
                       help="Output audio device index (use --list_devices to see options)")
    parser.add_argument("--duration", "-t", type=float, default=None,
                       help="Duration in seconds (default: run until interrupted)")
    parser.add_argument("--output_file", "-o", default=None,
                       help="Output file to save processed audio")
    parser.add_argument("--playback", "-p", action="store_true",
                       help="Enable real-time playback of processed audio")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with additional output information")
    parser.add_argument("--test_audio", action="store_true",
                       help="Test audio system by playing input directly (no processing)")
    parser.add_argument("--list_devices", action="store_true",
                       help="List available audio input devices")
    parser.add_argument("--block_size", "-b", type=int, default=128,
                       help="Audio block size for real-time processing")
    
    args = parser.parse_args()
    
    if args.list_devices:
        # Create a temporary AEC processor just for listing devices
        # We don't need to load models for device listing
        aec_processor = RealTimeAEC(
            model_path=None,
            block_size=args.block_size,
            input_device=args.device,
            output_device=args.output_device,
            load_models=False
        )
        aec_processor.list_audio_devices()
        return
    
    # For actual processing, model is required
    if not args.model:
        parser.error("--model/-m is required for audio processing (use --list_devices to see available devices)")
    
    # Display user-selected parameters
    print("USER SELECTED PARAMETERS:")
    print(f"  Model path: {args.model}")
    print(f"  Input device: {args.device if args.device is not None else 'Default'}")
    print(f"  Output device: {args.output_device if args.output_device is not None else 'Default'}")
    print(f"  Duration: {args.duration if args.duration is not None else 'Until interrupted'}")
    print(f"  Output file: {args.output_file if args.output_file is not None else 'None'}")
    print(f"  Playback enabled: {args.playback}")
    print(f"  Block size: {args.block_size}")
    print()
    
    # Create AEC processor
    aec_processor = RealTimeAEC(
        model_path=args.model,
        block_size=args.block_size,
        input_device=args.device,
        output_device=args.output_device
    )
    
    # Store debug flag and test mode
    aec_processor.debug_mode = args.debug
    aec_processor.test_mode = args.test_audio
    
    # Verify model files exist
    if not os.path.exists(args.model + "_1.tflite"):
        print(f"Error: Model file {args.model}_1.tflite not found!")
        return
    if not os.path.exists(args.model + "_2.tflite"):
        print(f"Error: Model file {args.model}_2.tflite not found!")
        return
    
    try:
        # Start processing
        aec_processor.start_processing(
            duration=args.duration,
            output_file=args.output_file,
            enable_playback=args.playback
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()