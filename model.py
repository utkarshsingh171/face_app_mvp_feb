import cv2
import numpy as np
import shutil
import os
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.45

# Initialize model once
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def find_faces(reference_img, group_images, output_folder="output"):

    os.makedirs(output_folder, exist_ok=True)

    faces = app.get(reference_img)
    if len(faces) == 0:
        return "No face found in reference image"

    ref_emb = faces[0].embedding
    ref_emb = ref_emb / np.linalg.norm(ref_emb)

    matched_images = []

    for i, img in enumerate(group_images):

        faces = app.get(img)

        for face in faces:

            face_emb = face.embedding
            face_emb = face_emb / np.linalg.norm(face_emb)

            similarity = np.dot(ref_emb, face_emb)

            if similarity > SIMILARITY_THRESHOLD:

                filename = f"match_{i}.jpg"
                path = os.path.join(output_folder, filename)

                cv2.imwrite(path, img)
                matched_images.append(path)

    return matched_images