import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.45

# -----------------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------------
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def find_faces(reference_imgs, group_images, operation="AND", output_folder="output"):

    os.makedirs(output_folder, exist_ok=True)

    ref_embeddings = []

    # -----------------------------
    # GET REFERENCE EMBEDDINGS
    # -----------------------------
    for ref_img in reference_imgs:

        faces = app.get(ref_img)

        if len(faces) == 0:
            return "No face found in one of the reference images"

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)

        ref_embeddings.append(emb)

    matched_images = []

    # -----------------------------
    # PROCESS GROUP IMAGES
    # -----------------------------
    for i, img in enumerate(group_images):

        faces = app.get(img)

        if len(faces) == 0:
            continue

        face_embeddings = []

        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            face_embeddings.append(emb)

        match_found = False

        # -----------------------------
        # AND LOGIC (ALL PERSONS)
        # -----------------------------
        if operation == "AND":

            all_matched = True

            for ref_emb in ref_embeddings:

                single_match = False

                for face_emb in face_embeddings:
                    similarity = np.dot(ref_emb, face_emb)

                    if similarity > SIMILARITY_THRESHOLD:
                        single_match = True
                        break

                if not single_match:
                    all_matched = False
                    break

            match_found = all_matched

        # -----------------------------
        # OR LOGIC (ANY PERSON)
        # -----------------------------
        elif operation == "OR":

            for ref_emb in ref_embeddings:
                for face_emb in face_embeddings:

                    similarity = np.dot(ref_emb, face_emb)

                    if similarity > SIMILARITY_THRESHOLD:
                        match_found = True
                        break

                if match_found:
                    break

        # -----------------------------
        # SAVE MATCHED IMAGE
        # -----------------------------
        if match_found:

            filename = f"match_{i}.jpg"
            path = os.path.join(output_folder, filename)

            # Save with high quality
            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            matched_images.append(path)

    return matched_images