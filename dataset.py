import os
import torch
from datasets import Dataset, Features, Value, Audio

# Define the data for Gujarati language
data = {
    "text": [
        "API શું છે?",  # What is an API?
        "RESTful સેવાઓને સમજાવો.",  # Explain RESTful services.
        "OAuth કેવી રીતે કાર્ય કરે છે?",  # How does OAuth work?
        "CUDA શું છે?"  # What is CUDA?
    ],
    "audio": [
        "D:\\Gujarati_TTS\\gujarati_audio\\audio1.wav",
        "D:\\Gujarati_TTS\\gujarati_audio\\audio2.wav",
        "D:\\Gujarati_TTS\\gujarati_audio\\audio3.wav",
        "D:\\Gujarati_TTS\\gujarati_audio\\audio4.wav"
    ]
}

# Define the structure of the dataset
features = Features({
    "text": Value("string"),
    "audio": Audio(sampling_rate=16000)
})

# Create the dataset
dataset = Dataset.from_dict(data, features=features)

# Verify the dataset
full_size = len(dataset)
# Select the full dataset
dataset = dataset.select(range(full_size))
print(dataset)

# Save the dataset to disk
dataset.save_to_disk("gujarati_tts_dataset")

print("Dataset saved to gujarati_tts_dataset")
