import base64
from io import BytesIO

import face_recognition
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont

# test_images = ["photos/anun_ryan_test.jpg"]

app = FastAPI()
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def recog(new_image):

    test_image = face_recognition.load_image_file(new_image)

    pil_image = Image.fromarray(np.asarray(test_image))

    draw = ImageDraw.Draw(pil_image)

    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    train_images = [
        "photos/James.png",
        "photos/Anun.png",
        "photos/Ryan.png",
    ]

    known_face_names = ["James", "Anun", "Ryan"]

    face_names = []

    encodings = []

    for train_image in train_images:
        train_image = face_recognition.load_image_file(train_image)
        train_encoding = face_recognition.face_encodings(train_image)[0]
        # print(train_image)
        # print(train_encoding)
        encodings.append(train_encoding)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(
            encodings, face_encoding, tolerance=0.45
        )

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_names.append(name)

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(
            ((left, bottom - text_height - 10), (right, bottom)),
            fill=(0, 0, 255),
            outline=(0, 0, 255),
        )
        font = ImageFont.truetype(r"./arial.ttf", 20)
        draw.text(
            (left + 6, bottom - text_height - 8),
            name,
            font=font,
            fill=(255, 255, 255, 255),
        )

    del draw

    # Display the resulting image
    # pil_image.show()

    # save image
    pil_image.save("photos/face_recog.jpg")

    # convert image to bytes and return image together with names

    # imgByteArr = BytesIO()
    # pil_image.save(imgByteArr, format='JPEG')
    # imgByteArr = imgByteArr.getvalue()

    # return face_names, FileResponse("photos/face_recog.jpg", media_type="image/jpeg")

    with open("photos/face_recog.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return JSONResponse(
        content={"names": face_names, "image": encoded_string.decode("utf-8")}
    )

    return FileResponse("photos/face_recog.jpg", media_type="image/jpeg")


@app.post("/upload")
async def create_file(file: bytes = File()):
    file_object = BytesIO(file)
    return recog(file_object)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
