import face_recognition

test_images = ["photos/anun_ryan_test.jpg"]

train_images = ["photos/james_test.jpg", "photos/anun_train.jpg", "photos/ryan_train2.jpg"]

known_face_names = ["James", "Anun", "Ryan"]

face_names = []

encodings = []

for train_image in train_images:
    train_image = face_recognition.load_image_file(train_image)
    train_encoding = face_recognition.face_encodings(train_image)[0]
    encodings.append(train_encoding)


test_image = face_recognition.load_image_file(test_images[0])
face_locations = face_recognition.face_locations(test_image)


for face_location in face_locations:
    face_encoding = face_recognition.face_encodings(test_image, [face_location])[0]
    matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.4)
    print(matches)
    if True in matches:
        index = matches.index(True)
        name = known_face_names[index]
        face_names.append(name)

print(face_names)



