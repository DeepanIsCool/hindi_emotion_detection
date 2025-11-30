# 🎙️ Hindi/Indian English Emotion Recognition

A Streamlit web application for detecting emotions from Hindi and Indian English speech audio files using a fine-tuned Wav2Vec2-XLSR-53 model.

## Features

- Upload `.wav` audio files for emotion analysis
- Real-time emotion detection with confidence scores
- Visual breakdown of emotion probabilities
- Support for 7 emotion classes:
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise

## Model

This application uses a fine-tuned **Wav2Vec2-XLSR-53** model trained on the IESC (Indian Emotional Speech Corpus) dataset for emotion classification.

### Model Architecture

- Base model: `Wav2Vec2ForSequenceClassification`
- Hidden size: 1024
- Attention heads: 16
- Hidden layers: 24

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DeepanIsCool/hindi_emotion_detection.git
cd hindi_emotion_detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- streamlit
- torch
- transformers
- librosa
- numpy
- soundfile

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`).

### How to use:

1. Upload a `.wav` audio file using the file uploader
2. Wait for the model to analyze the audio
3. View the detected emotion and confidence score
4. Check the detailed probability breakdown chart

## Project Structure

```
hindi_emotion_detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── hindi_emotion_model/      # Pre-trained model files
    ├── config.json           # Model configuration
    ├── model.safetensors     # Model weights
    └── preprocessor_config.json  # Feature extractor config
```

## Notes

- Audio files are automatically resampled to 16kHz
- Maximum audio length is capped at 4 seconds (as per training configuration)
- For best results, use clear speech recordings with minimal background noise

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Built with [Streamlit](https://streamlit.io/)
- Model based on [Wav2Vec2-XLSR-53](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
