import cv2
import numpy as np
from time import sleep
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('D:/Emotion_detection_with_CNN-main/Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:/Emotion_detection_with_CNN-main/Emotion_detection_with_CNN-main/model/emotion_model.h5")
print("Loaded model from disk")

# cap=cv2.imread("D:\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\data\test\angry\PrivateTest_731447.jpg")
# start the webcam feed
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("D:\\Emotion_detection_with_CNN-main\\Emotion_detection_with_CNN-main\\samplevideo.mp4")
frames = 10
emotions_detected = []
while frames > 0:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        emotions_detected.append(emotion_dict[maxindex])
        sleep(1)
        frames-=1

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(emotions_detected)
res = max(set(emotions_detected), key = emotions_detected.count)
print(res)

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('D:\Emotion_detection_with_CNN-main\Emotion_detection_with_CNN-main\data\movies_metadata.csv')
movies_data.dropna(inplace=True)
movies_data = movies_data.reset_index(drop=True)
movies_data=movies_data.drop(columns=["imdb_id"])
movies_data = movies_data[movies_data['genres'] != '[]']
for i, rows in movies_data.iterrows():
  a = rows['genres'].split(", ")
  k = []
  for j in range(1, len(a), 2):
    k.append(a[j])

  p = []
  for tag in k:
    p.append(tag[9:-1])

  h = []
  for b in p:
    h.append(b[0:len(b)-1])
  e = h[len(h)-1]
  h[len(h)-1] = e[0:len(e)-1]
  a = ','.join(h)
  print(i, " ", a)
  movies_data.at[i, 'genres'] = a
selected_features = ['genres','tagline']
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

if(res=='Happy'):
    genre_name='Action,Comedy,Family'
elif(res=='Sad'):
    genre_name='Comedy'
elif(res=='Neutral'):
    genre_name='Fantasy'
elif(res=='Surprised'):
    genre_name='Mystery'
# genre_name = input(' Enter genre type: ')
list_of_all_genres = movies_data['genres'].tolist()
find_close_match = difflib.get_close_matches(genre_name, list_of_all_genres)
for i, rows in movies_data.iterrows():
  movies_data.id=i;
close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.genres == close_match]['id'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

cap.release()
cv2.destroyAllWindows()
