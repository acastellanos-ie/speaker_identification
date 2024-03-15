import os
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
import numpy as np
import os

import streamlit as st
import tempfile
import os
import io

import ffmpeg

from pathlib import Path

natasa_path = Path("./data/natasa")
eddy_path = Path("./data/eddy")


def extract_audio_from_video(video_path, target_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(target_audio_path)
    video.close()

def split_audio_into_chunks(audio_path, segment_length=15):
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_segment = segment_length * sr
    total_segments = int(np.ceil(len(y) / samples_per_segment))

    for segment in range(total_segments):
        start_sample = segment * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment_data = y[start_sample:end_sample]
        
        segment_file_path = audio_path.replace('.wav', f'_chunk{segment}.wav')
        sf.write(segment_file_path, segment_data, sr)

def process_videos_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4'):  # Check for video files
            video_path = os.path.join(folder_path, file_name)
            target_audio_path = video_path.replace('.mp4', '.wav')
            
            # Extract audio from video
            extract_audio_from_video(video_path, target_audio_path)
            
            # Split the extracted audio and save chunks
            split_audio_into_chunks(target_audio_path, segment_length=15)
            
            # Optionally, remove the original extracted audio file if no longer needed
            os.remove(target_audio_path)

def extract_features(audio_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.

    Args:
    audio_path (str): Path to the audio file.
    n_mfcc (int): Number of MFCC features to extract.

    Returns:
    np.array: Extracted MFCC features.
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def prepare_dataset(folders):
    """
    Prepare the dataset from folders containing the audio files for each person.

    Args:
    folders (list of str): List containing paths to folders, one per person.

    Returns:
    X (np.array): The feature vectors extracted from the audio.
    y (np.array): The labels for each feature vector.
    """
    X, y = [], []
    label_to_int = dict()  # To dynamically assign labels based on folder names

    for folder_idx, folder_path in enumerate(folders):
        person_name = os.path.basename(folder_path)
        label_to_int[person_name] = folder_idx  # Assign a unique integer to each person
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                features = extract_features(file_path)
                X.append(features)
                y.append(folder_idx)  # Use the folder index as the label

    return np.array(X), np.array(y), label_to_int


def predict_speaker_from_video(uploaded_file, model, label_map, segment_length=15):
    """
    Predict the speaker from an uploaded video file using ffmpeg-python for audio extraction.

    Args:
    uploaded_file: The uploaded file object from Streamlit.
    model: The trained machine learning model for prediction.
    label_map: A dictionary mapping numerical labels to speaker names.
    segment_length: Length of each audio segment for processing, in seconds.

    Returns:
    The name of the predicted speaker.
    """
    # Use ffmpeg to extract audio directly from the uploaded video bytes
    try:
        out, _ = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav')
            .run(input=uploaded_file.getvalue(), capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('ffmpeg error:', e.stderr)
        raise e

    audio_bytes_io = io.BytesIO(out)  # Convert ffmpeg output to BytesIO for further processing

    # Load the audio with librosa directly from the BytesIO object
    y, sr = librosa.load(audio_bytes_io, sr=None)
    
    predictions = []
    samples_per_segment = segment_length * sr
    total_segments = int(np.ceil(len(y) / samples_per_segment))
    
    for segment in range(total_segments):
        start_sample = segment * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment_data = y[start_sample:end_sample]

        mfccs = librosa.feature.mfcc(y=segment_data, sr=sr, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)

        prediction = model.predict(mfccs_processed)
        predictions.append(prediction[0])
    
    final_prediction = max(set(predictions), key=predictions.count)
    speaker_name = label_map[final_prediction]
    
    return speaker_name

def main(model):
    st.title('Speaker Identification App')

    uploaded_file = st.file_uploader("Choose a video file...", type=['mp4'])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        # Assuming label_map is defined
        predicted_speaker = predict_speaker_from_video(uploaded_file, model, label_map)
            
        st.write(f"Predicted Speaker: {predicted_speaker}")


        
#process_videos_in_folder(natasa_path)
#process_videos_in_folder(eddy_path)

if __name__ == '__main__':
    folders = [Path("./data/natasa"), Path("./data/eddy")]
    X, y, label_map = prepare_dataset(folders)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    main(clf)



