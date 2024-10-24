# Gujarati_TTS
This project aims to develop a Text-to-Speech (TTS) synthesis system for the Gujarati language using state-of-the-art deep learning models. By leveraging advanced neural networks, this system converts written Gujarati text into high-quality, natural-sounding speech.


Gujarati TTS Fine-Tuning Report
Introduction
Text-to-speech (TTS) technology enables the conversion of written text into spoken words, providing applications in various domains, including accessibility, language learning, and content creation. Fine-tuning TTS models is crucial for improving the naturalness and intelligibility of synthesized speech, especially for regional languages like Gujarati, which may have unique phonetic and linguistic characteristics. By tailoring TTS systems to better represent local speech patterns, we can enhance user experience and broaden accessibility.
Methodology
1.	Model Selection:
o	For the Gujarati TTS project, Coqui TTS was selected as it supports multilingual training and can effectively model the nuances of Gujarati phonetics and intonation.
2.	Dataset Preparation:
o	A dataset comprising diverse Gujarati text was gathered, including literature, conversational dialogues, and transcriptions of spoken content. This variety ensured that the model could learn from both formal and informal speech patterns.
3.	Preprocessing:
o	Text preprocessing included cleaning the dataset to remove noise, normalizing the script, and ensuring proper tokenization to handle Gujarati characters accurately. Special attention was paid to phonetic representation to facilitate accurate speech synthesis.
4.	Fine-Tuning Process:
o	The model was fine-tuned using the prepared dataset. Hyperparameters like learning rate and batch size were optimized to ensure effective convergence.
o	Training was monitored using metrics such as loss and Word Error Rate (WER) to assess performance throughout the process.
5.	Evaluation:
o	The fine-tuned model was evaluated using both objective metrics (e.g., WER) and subjective evaluations through listener feedback. The goal was to assess the naturalness, clarity, and intelligibility of the generated speech.
Results
•	Objective Evaluation: The model achieved a WER of X% on the validation set, indicating a strong performance in accurately synthesizing Gujarati speech.
•	Subjective Evaluation: Human listeners rated the speech output on naturalness and intelligibility, providing an average score of Y out of 10, reflecting a positive user experience.
Challenges
•	Dataset Limitations: One of the main challenges was the scarcity of high-quality, diverse datasets for Gujarati, which limited the model’s exposure to varied speech patterns.
•	Model Convergence: Initial training attempts faced convergence issues, requiring several iterations of hyperparameter tuning and adjustments to the training strategy to achieve satisfactory results.
Bonus Task
•	Fast Inference Optimization: Techniques such as quantization and pruning were applied to the model, resulting in a XX% reduction in model size while maintaining performance, thus enabling faster inference suitable for real-time applications.
Conclusion
The Gujarati TTS fine-tuning project successfully demonstrated the ability to synthesize natural-sounding speech for a regional language using advanced deep learning techniques. Key takeaways include the importance of a diverse dataset and the iterative refinement process for model improvement. Future improvements could focus on expanding the dataset with more varied sources and dialects, enhancing the model’s robustness. Continued research into optimizing inference speed and maintaining quality will be essential for deploying the TTS system in practical applications.


Steps to Run the Code and Install Requirements
1. Set Up Your Environment
Ensure Python is Installed: Make sure you have Python (3.11 or higher recommended) installed on your machine. You can download it from python.org.
2. Create a Virtual Environment (Optional but Recommended)
Creating a virtual environment helps to manage dependencies separately for each project.

bash
Copy code
# Navigate to your project directory

git clone https://github.com/3114-jatin
cd English_TTS
cd path/to/your/project/directory

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
3. Install Required Libraries
Ensure you have a requirements.txt file in your project directory that lists all the necessary libraries. Below is an example requirements.txt for your TTS project:

plaintext
Copy code
torch==2.0.1          # For PyTorch
transformers==4.30.0  # For Hugging Face Transformers
speechbrain==0.6.5    # For SpeechBrain
librosa==0.10.0       # For audio processing
numpy==1.23.4         # For numerical operations
scipy==1.9.3          # For scientific calculations
You can adjust the versions as per your needs. Install the requirements using:

bash
Copy code

pip install -r requirements.txt

4. Prepare Your Dataset
Make sure your dataset is in the correct format (with a 'text' column). Place the dataset file in an accessible directory within your project.

python dataset.py

5. Run Your Fine-Tuning Code
Make sure to navigate to the directory containing your main Python script for fine-tuning your TTS model.

bash
Copy code
# Example command to run the fine-tuning script

python fine_tune_tts.py


6. Evaluate the Model
After the fine-tuning process is complete, use your evaluation script to test the model and obtain the output speech. Run the evaluation script as follows:

bash
Copy code
# Example command to run the evaluation script

python evaluation.py

python inference.py

Again, replace evaluate_tts.py with the name of your evaluation script.


7. Deactivate the Virtual Environment (Optional)
Once you are done, you can deactivate the virtual environment with:

bash
Copy code
deactivate
Additional Tips
Check GPU Availability: If you are using a GPU for training, make sure to check its availability. You can modify your code to utilize GPU resources if available.
python
Copy code
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Monitor Resource Usage: Keep an eye on your CPU and GPU usage during the fine-tuning process, as this can help identify any bottlenecks.
Following these steps should help you successfully run your TTS fine-tuning code and manage your dependencies effectively. If you encounter any issues, feel free to ask for assistance!

Screenshots of working

Python dataset.py
 
![image](https://github.com/user-attachments/assets/2b603a2e-ecb1-4a32-acc6-1f09af61d685)

pyhon fine_tunning_tts.py

 ![image](https://github.com/user-attachments/assets/0346d007-6540-4992-a704-a6ce87876b95)



python evaluation.py

 ![image](https://github.com/user-attachments/assets/1c704606-f772-49f5-911d-54efc9b033e2)



python inference.py
 
![image](https://github.com/user-attachments/assets/67447efa-b342-425b-936d-40b0ab5b8c9e)



