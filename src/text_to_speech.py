"""
Text-to-Speech Module using OpenAI TTS API
Converts recognized text to natural-sounding speech
"""

import os
import random
from pathlib import Path
from typing import Optional
import pygame
from openai import OpenAI
from datetime import datetime
import tempfile


class TextToSpeech:
    """
    Text-to-Speech conversion using OpenAI TTS API
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy",
        output_dir: Optional[str] = None
    ):
        """
        Initialize TextToSpeech
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: TTS model to use ('tts-1' or 'tts-1-hd')
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            output_dir: Directory to save audio files (if None, uses temp dir)
        """
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Settings
        self.model = model
        self.voice = voice
        self.output_dir = output_dir or tempfile.gettempdir()
        
        # Create output directory only if it's not a temp directory
        if output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # State
        self.is_speaking = False
        self.current_audio_file = None
        
        print(f"TextToSpeech initialized with model: {model}, voice: {voice}")
    
    def text_to_speech(
        self,
        text: str,
        save_file: bool = False,
        custom_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
            save_file: Whether to save the audio file permanently
            custom_filename: Custom filename for saved audio
            
        Returns:
            Path to audio file if saved, None otherwise
        """
        if not text or len(text.strip()) == 0:
            print("No text to convert to speech")
            return None
        
        try:
            print(f"Converting to speech: '{text}'")
            
            # Generate speech
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
            
            # Generate filename
            if custom_filename:
                filename = custom_filename
            elif save_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tts_{timestamp}.mp3"
            else:
                # Use unique temp filename to avoid conflicts
                random_id = random.randint(1000, 9999)
                filename = f"temp_tts_{random_id}.mp3"
            
            # Full path
            audio_path = os.path.join(self.output_dir, filename)
            
            # Save audio file
            response.stream_to_file(audio_path)
            
            print(f"Audio saved to: {audio_path}")
            
            # Play audio
            self.play_audio(audio_path)
            
            # Clean up temp file if not saving
            if not save_file:
                self.current_audio_file = audio_path
            
            return audio_path if save_file else None
            
        except Exception as e:
            print(f"Error in text_to_speech: {e}")
            return None
    
    def play_audio(self, audio_path: str):
        """
        Play audio file
        
        Args:
            audio_path: Path to audio file
        """
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return
            
            # Stop and unload previous audio to release file lock
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            self.is_speaking = True
            
            # Load and play audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            print("Playing audio...")
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_speaking = False
    
    def stop_audio(self):
        """Stop currently playing audio"""
        try:
            pygame.mixer.music.stop()
            self.is_speaking = False
            print("Audio stopped")
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def is_playing(self) -> bool:
        """
        Check if audio is currently playing
        
        Returns:
            True if audio is playing, False otherwise
        """
        return pygame.mixer.music.get_busy()
    
    def wait_for_completion(self):
        """Wait for current audio to finish playing"""
        while self.is_playing():
            pygame.time.Clock().tick(10)
    
    def set_volume(self, volume: float):
        """
        Set playback volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(volume)
        print(f"Volume set to {volume}")
    
    def change_voice(self, voice: str):
        """
        Change TTS voice
        
        Args:
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
        """
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if voice in valid_voices:
            self.voice = voice
            print(f"Voice changed to: {voice}")
        else:
            print(f"Invalid voice. Choose from: {valid_voices}")
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Stop any playing audio
            self.stop_audio()
            
            # Delete temporary audio file
            if self.current_audio_file and os.path.exists(self.current_audio_file):
                os.remove(self.current_audio_file)
                print(f"Temporary file deleted: {self.current_audio_file}")
            
            # Quit pygame mixer
            pygame.mixer.quit()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def batch_convert(self, texts: list, output_prefix: str = "batch"):
        """
        Convert multiple texts to speech files
        
        Args:
            texts: List of text strings
            output_prefix: Prefix for output filenames
            
        Returns:
            List of generated audio file paths
        """
        audio_files = []
        
        for i, text in enumerate(texts):
            filename = f"{output_prefix}_{i+1:03d}.mp3"
            audio_path = self.text_to_speech(text, save_file=True, custom_filename=filename)
            if audio_path:
                audio_files.append(audio_path)
        
        print(f"Batch conversion complete: {len(audio_files)} files created")
        return audio_files


class SpeechBuffer:
    """
    Buffer for accumulating recognized gestures before converting to speech
    """
    
    def __init__(self, tts_engine: TextToSpeech, min_word_length: int = 3):
        """
        Initialize SpeechBuffer
        
        Args:
            tts_engine: TextToSpeech instance
            min_word_length: Minimum word length before triggering speech
        """
        self.tts = tts_engine
        self.min_word_length = min_word_length
        self.buffer = []
        self.current_word = ""
    
    def add_character(self, char: str):
        """
        Add character to buffer
        
        Args:
            char: Character to add
        """
        if char and len(char) == 1:
            self.current_word += char
            print(f"Current word: {self.current_word}")
    
    def add_space(self):
        """Add space (complete current word)"""
        if self.current_word:
            self.buffer.append(self.current_word)
            print(f"Word completed: {self.current_word}")
            self.current_word = ""
    
    def delete_last_character(self):
        """Delete last character from current word"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            print(f"Current word: {self.current_word}")
    
    def clear(self):
        """Clear buffer and current word"""
        self.buffer = []
        self.current_word = ""
        print("Buffer cleared")
    
    def get_text(self) -> str:
        """
        Get complete text from buffer
        
        Returns:
            Complete text string
        """
        full_text = " ".join(self.buffer)
        if self.current_word:
            full_text += " " + self.current_word if full_text else self.current_word
        return full_text.strip()
    
    def speak_current(self):
        """Speak the current accumulated text"""
        text = self.get_text()
        if text:
            self.tts.text_to_speech(text)
    
    def speak_and_clear(self):
        """Speak current text and clear buffer"""
        text = self.get_text()
        if text:
            self.tts.text_to_speech(text)
            self.clear()


# Example usage and testing
def main():
    """Demo function to test TextToSpeech"""
    print("TextToSpeech Demo")
    print("-" * 50)
    
    # Initialize TTS (make sure OPENAI_API_KEY is set in environment)
    try:
        tts = TextToSpeech(voice="alloy")
        
        # Test basic TTS
        test_texts = [
            "Hello, this is a test of the text to speech system.",
            "I can help people with hearing impairments communicate.",
            "This uses OpenAI's advanced text to speech technology."
        ]
        
        for text in test_texts:
            print(f"\nSpeaking: {text}")
            tts.text_to_speech(text)
            tts.wait_for_completion()
        
        # Test SpeechBuffer
        print("\n" + "=" * 50)
        print("Testing SpeechBuffer...")
        buffer = SpeechBuffer(tts)
        
        # Simulate spelling out a word
        word = "HELLO"
        for char in word:
            buffer.add_character(char)
        
        buffer.add_space()
        
        word2 = "WORLD"
        for char in word2:
            buffer.add_character(char)
        
        print(f"\nFinal text: {buffer.get_text()}")
        buffer.speak_current()
        
        # Cleanup
        tts.cleanup()
        
        print("\nDemo complete!")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    main()
