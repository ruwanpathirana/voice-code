from flask import Flask, request, render_template
import os
from class_emotion import EmotionDetector
from flask import jsonify
TF_ENABLE_ONEDNN_OPTS=0

app = Flask(__name__)

@app.route('/emo', methods=['POST'])
def upload_file():

    if not os.path.exists('cache'):
        os.makedirs('cache')

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = file.filename
        file_path = os.path.join('cache', filename)
        file.save(file_path)
        
    
    uploaded_file_path = file_path
    emotion_detector = EmotionDetector(uploaded_file_path)
    emotions = emotion_detector.test_uploaded_audio()

    return emotions, 200  # Return emotions as JSON response


if __name__ == '__main__':
    app.run(debug=True)
