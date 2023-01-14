from io import BytesIO

import face_recognition
import uvicorn
from fastapi import FastAPI, File, UploadFile

# test_images = ["photos/anun_ryan_test.jpg"]



app = FastAPI()

def recog(new_image):

    
    test_image = face_recognition.load_image_file(new_image)
    face_locations = face_recognition.face_locations(test_image)

    # tell me how many faces in the image
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    train_images = ["photos/james_test.jpg", "photos/anun_train.jpg", "photos/ryan_train2.jpg"]

    known_face_names = ["James", "Anun", "Ryan"]

    face_names = []

    encodings = []

    for train_image in train_images:
        train_image = face_recognition.load_image_file(train_image)
        train_encoding = face_recognition.face_encodings(train_image)[0]
        # print(train_image)
        # print(train_encoding)
        encodings.append(train_encoding)

    for face_location in face_locations:
        face_encoding = face_recognition.face_encodings(test_image, [face_location])[0]
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.4)
        print(matches)
        if True in matches:
            index = matches.index(True)
            name = known_face_names[index]
            face_names.append(name)
    print(face_names)

    return face_names

@app.post("/upload")
async def create_file(file: bytes = File()):
    file_object = BytesIO(file)
    return recog(file_object)
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
















