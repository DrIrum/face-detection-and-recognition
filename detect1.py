import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
shahrukh_image = face_recognition.load_image_file("shahrukh.jpeg")
shahrukh_face_encoding = face_recognition.face_encodings(shahrukh_image)[0]

# Load a second sample picture and learn how to recognize it.
rock_image = face_recognition.load_image_file("rock.jpeg")
rock_face_encoding = face_recognition.face_encodings(rock_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    shahrukh_face_encoding,
    rock_face_encoding
]
known_face_names = [
    "Shahrukh Khan",
    "The Rock"
]

unknown_image = face_recognition.load_image_file("john.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

pil_image = Image.fromarray(unknown_image)

draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

del draw

pil_image.show()
