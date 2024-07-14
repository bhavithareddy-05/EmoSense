import cv2
from deepface import DeepFace
from google.colab.patches import cv2_imshow
from collections import Counter
from multiprocessing import Pool
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from transformers import pipeline
import matplotlib.pyplot as plt

def process_frame(frame):
    emotions = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

    if emotions:
        return emotions[0]["dominant_emotion"]
    else:
        return None

def process_video(video_path):
    frame_list = []
    emotions_list = []
    capture = cv2.VideoCapture(video_path)
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray_frame, 1.1, 5)

        for (x, y, w, h) in faces:
            emotions = DeepFace.analyze(frame[y:y+h, x:x+w], actions=["emotion"], enforce_detection=False)

            if emotions:
                dominant_emotion = emotions[0]["dominant_emotion"]
                emotions_list.append(dominant_emotion)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            frame_list.append(frame)

    capture.release()
    emotion_counts = Counter(emotions_list)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    return frame_list, emotions_list, most_common_emotion

def transcribe_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    output_audio_path = "opaudio.mp3"
    audio_clip.write_audiofile(output_audio_path, codec='mp3', ffmpeg_params=["-ar", "16000", "-ac", "1"])
    video_clip.close()
    audio_clip.close()
    sound = AudioSegment.from_mp3(output_audio_path)
    sound.export("transcript.wav", format="wav")
    AUDIO_FILE = "transcript.wav"
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
        text = r.recognize_google(audio)
    return text

def emotion_detection(text):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def emotion_sentiment(text):
        results = classifier(text, padding='max_length', max_length=512)
        return {label['label']: label['score'] for label in results[0]}
    result = emotion_sentiment(text)
    return result

def display_emotion_results(results):
    emotions = list(results.keys())
    scores = list(results.values())
    plt.figure(figsize=(10, 6))
    plt.plot(emotions, scores, marker='o', linestyle='-')
    plt.title('Emotion Classification Results')
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print("Welcome to the Emotion Classifier!")
    print("Processing video...")
    frame_list, emotions_list, emotion_result = process_video(video_path)
    print(f"Most common emotion from video: {emotion_result}")
    for frame, emotion in zip(frame_list, emotions_list):
        if emotion == emotion_result:
            cv2.putText(frame, str(emotion_result), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2_imshow(frame)
            break

    print("Transcribing audio...")
    text_result = transcribe_audio(video_path)
    print(f"Transcription: {text_result}")
    print("Detecting emotion from transcription...")
    results = emotion_detection(text_result)
    display_emotion_results(results)

if __name__ == "__main__":
    video_path = "/content/video-1.mp4"
    main()
