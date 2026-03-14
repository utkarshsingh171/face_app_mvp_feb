import streamlit as st
import numpy as np
import cv2
from PIL import Image
from model import find_faces

st.set_page_config(page_title="Face Finder", layout="wide")

st.title("🔎 Face Finder")

st.write("Upload a reference face and multiple group photos")

# Upload reference image
ref_file = st.file_uploader(
    "Upload Reference Face",
    type=["jpg", "jpeg", "png"]
)

# ---------- SHOW REFERENCE ----------

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

# ---------- SHOW GROUP PHOTOS ----------
if ref_file and group_files:

    st.subheader("Group Photos")

    group_images = []

    cols = st.columns(4)   # number of images per row

    for i, file in enumerate(group_files):

        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        group_images.append(img_np)

        with cols[i % 4]:
            st.image(img, caption=file.name, width=300)

# ---------- PROCESS ----------
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