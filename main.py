import threading
import face_recognition
import cv2
import os
import numpy as np
from playsound import playsound
import time

count = 0
keepname = ""

video_capture = cv2.VideoCapture(0)

train_dir = os.listdir('/Users/kyrieyang/Desktop/Greeting-software/Known/')

known_face_encodings = []

dick_head = {}

known_face_names = []

for person in train_dir:
    if person == ".DS_Store":
        continue
    face = face_recognition.load_image_file("/Users/kyrieyang/Desktop/Greeting-software/Known/" + person)
    face_enc = face_recognition.face_encodings(face)[0]

    known_face_encodings.append(face_enc)
    known_face_names.append(person[:-4])
    dick_head.update({person[:-4] : 0})

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def greeting():
    while True:
        if keepname in known_face_names:
            if dick_head[keepname] == 0:
                try:
                    dick_head[keepname] = 1
                    playsound(keepname + '.m4a')
                except:
                    print("can't speak " + keepname)
        if keepname == "stop":
            break

greetingThread = threading.Thread(target=greeting)
greetingThread.start()

#print(dick_head)
#print(known_face_names)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        prob_order = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.45)
            name = "Unknown"
            prob = 1.0

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                 first_match_index = matches.index(True)
                 name = known_face_names[first_match_index]
                 prob = face_recognition.face_distance(known_face_encodings, face_encoding)[first_match_index]
            """
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                prob = face_recognition.face_distance(known_face_encodings, face_encoding)[first_match_index]
            """
            face_names.append(name)
            prob_order.append(prob)
            if name != "Unknown":
                keepname = name
                #print(keepname)

    process_this_frame = not process_this_frame

    i = 0
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, str(round((1-prob_order[i])*100,2)) + "%", (right, bottom - 6), font, 1.0, (255, 255, 255), 1)
        i += 1
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        keepname = "stop"
        time.sleep(1)
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
