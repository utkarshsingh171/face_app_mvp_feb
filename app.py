import streamlit as st
import numpy as np
import cv2
import zipfile
import io
import os
from model import find_faces

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Face Finder", layout="wide")

st.title("🔎 Face Finder")

# -----------------------------
# MODE SELECTION
# -----------------------------
mode = st.radio(
    "Mode",
    ["Single", "Multiple"],
    horizontal=True
)

# -----------------------------
# OPERATION (ONLY FOR MULTIPLE)
# -----------------------------
operation = "AND"

if mode == "Multiple":
    operation = st.radio(
        "Matching Condition",
        ["ALL (AND)", "ANY (OR)"],
        horizontal=True
    )

    operation = "AND" if operation == "ALL (AND)" else "OR"

# -----------------------------
# IMAGE LOADER
# -----------------------------
def load_image(file):
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_np

# -----------------------------
# REFERENCE UPLOAD
# -----------------------------
if mode == "Single":
    ref_files = st.file_uploader(
        "Reference Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    ref_files = [ref_files] if ref_files else []
else:
    ref_files = st.file_uploader(
        "Reference Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

# -----------------------------
# SHOW REFERENCE IMAGES
# -----------------------------
ref_images_np = []

if ref_files:
    st.subheader("Reference")

    cols = st.columns(min(4, len(ref_files)))

    for i, file in enumerate(ref_files):
        img_np = load_image(file)
        ref_images_np.append(img_np)

        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        with cols[i % len(cols)]:
            st.image(img_rgb, width=250)

# -----------------------------
# GROUP IMAGE UPLOAD
# -----------------------------
group_files = st.file_uploader(
    "Group Images",
    type=["jpg", "jpeg", "png","webp"],
    accept_multiple_files=True
)

# -----------------------------
# SHOW GROUP IMAGES
# -----------------------------
group_images = []

if group_files:
    st.subheader("Group Photos")

    cols = st.columns(min(4, len(group_files)))

    for i, file in enumerate(group_files):
        img_np = load_image(file)
        group_images.append(img_np)

        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        with cols[i % len(cols)]:
            st.image(img_rgb, width=300)

# -----------------------------
# FIND MATCHES
# -----------------------------
if ref_images_np and group_images:

    if st.button("Find Faces"):

        with st.spinner("Processing..."):

            # 🔥 IMPORTANT: pass operation
            results = find_faces(ref_images_np, group_images, operation=operation)

        if isinstance(results, str):
            st.error(results)

        elif len(results) == 0:
            st.warning("No matches found")

        else:
            st.success(f"{len(results)} matches found")

            cols = st.columns(3)

            for i, img_path in enumerate(results):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                with cols[i % 3]:
                    st.image(img_rgb, width=400)

            # ZIP download
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for img_path in results:
                    zip_file.write(img_path, os.path.basename(img_path))

            zip_buffer.seek(0)

            st.download_button(
                label="Download ZIP",
                data=zip_buffer,
                file_name="matched_faces.zip",
                mime="application/zip"
            )