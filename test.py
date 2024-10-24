from gtts import gTTS
from pydub import AudioSegment
import os

# List of Gujarati texts
texts = [
    "API શું છે?",  # What is an API?
    "RESTful સેવાઓને સમજાવો.",  # Explain RESTful services.
    "OAuth કેવી રીતે કાર્ય કરે છે?",  # How does OAuth work?
    "CUDA શું છે?"  # What is CUDA?
]

# Create a directory to store the audio files
output_dir = "gujarati_audio_wav"
os.makedirs(output_dir, exist_ok=True)

# Iterate over the texts and generate WAV audio files
for i, text in enumerate(texts):
    # Generate MP3 with gTTS
    tts = gTTS(text, lang='gu')
    mp3_path = os.path.join(output_dir, f"audio_{i+1}.mp3")
    tts.save(mp3_path)
    
    # Convert MP3 to WAV
    mp3_audio = AudioSegment.from_mp3(mp3_path)
    wav_path = os.path.join(output_dir, f"audio_{i+1}.wav")
    mp3_audio.export(wav_path, format="wav")
    
    # Optionally, remove the intermediate MP3 files
    os.remove(mp3_path)

    print(f"Saved WAV file: {wav_path}")

print("All WAV audio files have been generated successfully!")
