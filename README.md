# PerfectPitch.ai

A machine learning-based pitch detection system for identifying musical notes from audio.

Note: This version of the project is UNBUILT. I am leaving this publicly available for anyone
who wishes to experiment and build the model from scratch. If you are just interested in experimenting
with the model, a version of it will be released on PyPI in the near future.

## Overview

PerfectPitch.ai is a comprehensive project that demonstrates how to build a neural network for musical pitch detection. The system can identify the pitch class (C, C#/Db, D, etc.) from audio recordings in real-time. This project provides a complete pipeline from data generation to model training and interactive testing.

## Project Structure

- **Data Generation**: Creates MIDI files and converts them to audio for training
- **Feature Extraction**: Processes audio into chromagram representations
- **Model Training**: Trains a convolutional neural network for pitch classification
- **Real-time Testing**: Interactive interface for testing with microphone input

## Machine Learning Model

The project uses a Convolutional Neural Network (CNN) with the following architecture:

```
Input (12 x 600 x 1) → Conv2D(32, 3x3) → MaxPool(2x2) → Dropout(0.25) → 
Flatten → Dense(128) → Dropout(0.5) → Dense(12, softmax)
```

### Why CNN vs. MLP?

A CNN was chosen over a standard Multilayer Perceptron for several reasons:

1. **Spatial structure detection**: CNNs excel at capturing local patterns and relationships in the chromagram's pitch-time representation
2. **Parameter efficiency**: For the input shape (12 x 600 x 1), a CNN requires far fewer parameters while maintaining expressivity
3. **Translation invariance**: Musical patterns might appear at different time points, and CNNs can detect them regardless of timing
4. **Robustness to variations**: CNNs handle variations in amplitude, timing, and timbre better than MLPs

## Training Process

The model is trained using:
- **Loss function**: Sparse Categorical Cross-Entropy
- **Optimizer**: Adam
- **Batch size**: 64
- **Epochs**: 10

The training data is split 80/20 (training/testing) and the model weights are saved for later use in the real-time application.

## Data Generation

The training data is generated through a three-step process:

1. **MIDI Generation**: Creates MIDI files for all 12 pitch classes across 4 octaves using 128 different MIDI instruments
2. **Audio Conversion**: Converts MIDI files to WAV format using FluidSynth with a soundfont
3. **Feature Extraction**: Processes audio files to extract chromagram features and create the training dataset

This synthetic data approach ensures a large, diverse, and perfectly labeled dataset.

## Audio Processing

In the demo application, audio input is processed as follows:

1. Record 5 seconds of audio at 22050 Hz
2. Create a chromagram using librosa's `chroma_stft` function
3. Pad the chromagram to ensure consistent dimensions
4. Add batch and channel dimensions to match the model's input shape
5. Feed the processed chromagram to the model for prediction

A chromagram represents the energy distribution across the 12 pitch classes over time, making it an ideal feature for pitch detection.

## Installation and Usage

### Prerequisites
- Python 3.x
- FluidSynth
- SoundFont (.sf2) file

### Installation

1. Clone the repository
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Run the setup script
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Create necessary directories
   ```bash
   mkdir -p data/midifiles data/audio models logs
   ```

### Building the Network

1. Generate MIDI files
   ```bash
   python generate_midis.py
   ```

2. Convert MIDI to audio
   ```bash
   python generate_audio.py
   ```

3. Generate the dataset
   ```bash
   python generate_data.py
   ```

4. Train the model
   ```bash
   python train_model.py
   ```

### Using the Application

1. Start the real-time testing application
   ```bash
   python test_model.py
   ```

2. Press spacebar to record 5 seconds of audio
3. The model will predict the pitch class
4. Press ESC to exit

Add the `-c` flag to visualize the chromagram:
```bash
python test_model.py -c
```

## Troubleshooting

- **Audio recording issues**: Ensure your microphone is properly connected
- **FluidSynth errors**: Verify FluidSynth installation and SoundFont file
- **Model accuracy issues**: Try increasing training epochs or ensuring clear note input
- **Memory errors**: Reduce batch size or train on a subset of the data

## Future Improvements

Potential enhancements for this project could include:
- Chord detection (multiple pitches simultaneously)
- Octave detection in addition to pitch class
- Real-time continuous analysis rather than fixed 5-second recordings
- Integration with other musical software through MIDI or OSC

## License

- Licensed under GNU GPL 3 (see [LICENSE](LICENSE))
- This software is attributable to Aiden McCormack

## Acknowledgments

- Librosa for audio processing
- TensorFlow for machine learning
- FluidSynth for MIDI synthesis