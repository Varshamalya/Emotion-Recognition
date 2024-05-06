from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')
cam = cv2.VideoCapture(0)  # Initializing Camera

while True:
    success, frame = cam.read()
    print(success)
    frame = er.recognise_emotion(frame, return_type='BGR')
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    print(key)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
