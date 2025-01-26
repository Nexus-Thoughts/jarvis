import cv2
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from deepface import DeepFace

engine = pyttsx3.init()
recognizer = sr.Recognizer()
classifier = pipeline("sentiment-analysis")

def speak(text):
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."

def facial_emotion():
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    cv2.imwrite("face.jpg", frame)
    video.release()
    try:
        result = DeepFace.analyze(img_path="face.jpg", actions=['emotion'], enforce_detection=False)
        return result['dominant_emotion']
    except:
        return "Neutral"

def process_command(command):
    if "how are you" in command:
        emotion = facial_emotion()
        if emotion == "happy":
            speak("You look happy! What's the good news?")
        elif emotion == "sad":
            speak("You seem down. Want to talk about it?")
        else:
            speak("I'm good, thank you! How can I assist you?")
    elif "play music" in command:
        speak("Playing music now.")
    elif "next slide" in command:
        speak("Switching to the next slide.")
    elif "stop" in command:
        speak("Goodbye!")
        exit()
    else:
        response = classifier(command)
        speak(f"I think you're feeling {response[0]['label']}.")

while True:
    speak("Listening...")
    user_command = recognize_speech().lower()
    process_command(user_command)
