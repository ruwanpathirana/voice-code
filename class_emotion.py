import os
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from librosa.util.exceptions import ParameterError
import warnings
from pydub import AudioSegment
import json
warnings.filterwarnings("ignore")


class EmotionDetector:
    def __init__(self,file_path):
        # Load the model
        self.file_path = file_path
        model_path = 'model/emotion-recognition_NEW.hdf5'
        encoder_path = 'encode/encoder.pkl'
        self.model = load_model(model_path)

        # Load the encoder
        self.encoder = joblib.load(encoder_path)

    def extract_features(self, data, sample_rate):
        # zero crossing rate
        result = np.array([])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        if zcr.ndim > 1:
            zcr = zcr.flatten()
        result = np.hstack((result, zcr))

        # chroma shift
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        if chroma_stft.ndim > 1:
            chroma_stft = chroma_stft.flatten()
        result = np.hstack((result, chroma_stft))

        # mfcc
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        if mfcc.ndim > 1:
            mfcc = mfcc.flatten()
        result = np.hstack((result, mfcc))

        # rmse
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        if rms.ndim > 1:
            rms = rms.flatten()
        result = np.hstack((result, rms))

        # melspectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        if mel.ndim > 1:
            mel = mel.flatten()
        result = np.hstack((result, mel))

        # rollof
        rollof = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
        if rollof.ndim > 1:
            rollof = rollof.flatten()
        result = np.hstack((result, rollof))

        # centroids
        centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
        if centroid.ndim > 1:
            centroid = centroid.flatten()
        result = np.hstack((result, centroid))

        # contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
        if contrast.ndim > 1:
            contrast = contrast.flatten()
        result = np.hstack((result, contrast))

        # bandwidth
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
        if bandwidth.ndim > 1:
            bandwidth = bandwidth.flatten()
        result = np.hstack((result, bandwidth))

        # tonnetz
        tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
        if tonnetz.ndim > 1:
            tonnetz = tonnetz.flatten()
        result = np.hstack((result, tonnetz))

        return result

    def noise(self, data):
        noise_amp = 0.035 * np.random.uniform() * np.amax(data)
        stereo_noise = noise_amp * np.random.normal(size=data.shape)
        data_with_noise = data + stereo_noise
        return data_with_noise

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate=rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

    def get_features_recorded(self, data, sr):
        flag = 0
        if len(data) <= 1:
            print("Input signal length is too short for processing.")
            flag = 1
            return None, flag

        #get features for recorded audio using microphone
        res1 = self.extract_features(data, sr)
        result = np.array(res1)

        #get audio features with noise
        noise_data = self.noise(data)
        res2 = self.extract_features(noise_data, sr)
        result = np.vstack((result, res2))

        #get audio features with stretching and pitching
        new_data = self.stretch(data)
        data_stretch_pitch = self.pitch(new_data, sr)
        res3 = self.extract_features(data_stretch_pitch, sr)
        result = np.vstack((result, res3))

        return result, flag

    def extract_acoustic_features(self, audio_file, frame_length=2048, hop_length=512):
        y, sr = librosa.load(audio_file)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

        features = np.nan_to_num(features)

        return features.T

    def detect_sentence_change_points(self, audio_file, num_clusters=2):
        features = self.extract_acoustic_features(audio_file)
        kmeans = KMeans(n_clusters=num_clusters)
        predictions = kmeans.fit_predict(features)

        change_points = []
        prev_label = predictions[0]
        for i, label in enumerate(predictions[1:], start=1):
            if label != prev_label:
                change_points.append(i)
                prev_label = label

        return change_points

    def test_uploaded_audio(self):
        flag = 0
        emotions = []   # Store predicted emotions for each segment
        emotions_dict = {}

        # Check if the file format is wav, if not, convert to wav
        if self.file_path.lower().endswith('.wav'):
            converted_file_path = self.file_path
        else:
            # Convert to wav
            audio = AudioSegment.from_file(self.file_path)
            converted_file_path = os.path.splitext(self.file_path)[0] + '.wav'
            audio.export(converted_file_path, format='wav')

        # Load the uploaded audio file (either original wav or converted wav)
        audio, sr = librosa.load(converted_file_path, sr=None)
        duration = int(librosa.get_duration(y=audio, sr=sr))
        print("Audio Duration:", duration)

        # Check if the input signal length is sufficient for CQT
        if len(audio) <= 1:
            print("Input signal length is too short for processing")

        # Detect sentence change points
        try:
            change_points = self.detect_sentence_change_points(converted_file_path)
        except ParameterError as e:
            print("Error:", e)
            print("Input signal length is too short for CQT.")
            return

        # Iterate over segments and analyze emotion
        for i in range(duration):
            if i == 0:
                start_idx = 0
            else:
                start_idx = change_points[i-1]
            if i == len(change_points):
                end_idx = len(audio)
                break
            else:
                end_idx = change_points[i]

            segment = audio[start_idx:end_idx]
            feature, flag = self.get_features_recorded(segment, sr)
            print(flag)

            if flag == 0:
                # Apply min-max scaling
                scaler = MinMaxScaler()
                feature = scaler.fit_transform(feature)

                # Reshape the feature to match the input shape of the model
                feature = np.expand_dims(feature, axis=-1)

                # Get the predicted label
                label = self.model.predict(feature)

                # Reverse one hot encoded output to get the label information
                label_predicted = self.encoder.inverse_transform(label)
                label_strings = [str(label) for label in label_predicted[0]]

                emotions_dict[i+1] = label_strings

                emotions.append(label_predicted[0])  # Store predicted emotion for this segment

                print("\nEmotion Predicted For Segment {}: {}".format(i+1, label_predicted[0]))

            elif flag ==1:
                break

        # Flatten the nested list of emotions
        flat_emotions = [item for sublist in emotions for item in sublist]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(flat_emotions)
        plt.xlabel('Segment')
        plt.ylabel('Emotion')
        plt.title('Emotion Changes Over Segments')
        plt.grid(True)
        
        # Save the plot directly into a folder named 'graphs'
        if not os.path.exists('Graphs'):
            os.makedirs('Graphs')
        plt.savefig(os.path.join('graphs', 'emotion_plot.png'))
        plt.close()  # Close the plot to prevent it from displaying
        
        print("Final Output: ", emotions_dict)
        emotions_json = json.dumps(emotions_dict)
        return emotions_json





