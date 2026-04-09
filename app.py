import streamlit as st
import numpy as np
import cv2
import zipfile
import io
import os
from PIL import Image
from model import find_faces


st.set_page_config(page_title="Face Finder", layout="wide")

st.title("🔎 Face Finder")

st.write("Upload a reference face and multiple group photos")
st.write("Can upload only 1 reference photo")

ref_file = st.file_uploader(
    "Upload Reference Face",
    type=["jpg", "jpeg", "png"]
)


if ref_file:

    ref_img = Image.open(ref_file).convert("RGB")
    ref_img_np = np.array(ref_img)

    st.subheader("Reference Face")
    st.image(ref_img, width=300)


# Upload group photos
group_files = st.file_uploader(
    "Upload Group Photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if ref_file and group_files:

    st.subheader("Group Photos")

    group_images = []

    cols = st.columns(4)

    for i, file in enumerate(group_files):

        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        group_images.append(img_np)

        with cols[i % 4]:
            st.image(img, caption=file.name, width=300)

    if st.button("Find Face"):

        with st.spinner("Searching for matching faces..."):

            results = find_faces(ref_img_np, group_images)

        if isinstance(results, str):
            st.error(results)

        elif len(results) == 0:
            st.warning("No matches found")

        else:

            st.success(f"{len(results)} matches found")

            st.subheader("Matched Photos")

            cols = st.columns(3)

            for i, img_path in enumerate(results):

                img = cv2.imread(img_path)

                with cols[i % 3]:
                    st.image(img, width=600)

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for img_path in results:
                    zip_file.write(img_path, os.path.basename(img_path))

            zip_buffer.seek(0)

            st.download_button(
                label="⬇ Download All Matched Photos (ZIP)",
                data=zip_buffer,
                file_name="matched_faces.zip",
                mime="application/zip"
            )