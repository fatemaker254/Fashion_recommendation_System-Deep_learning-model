import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.set_page_config(layout="wide")  # Set wide layout

# Sidebar for file upload
st.sidebar.title("Fashion Recommender System")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# Display images without resizing
def display_image(image_path, caption):
    img = Image.open(image_path).convert("RGB")
    st.image(img, caption=caption, width=150)


# Main content
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # feature extract
        features = feature_extraction(
            os.path.join("uploads", uploaded_file.name), model
        )

        # recommendention
        indices = recommend(features, feature_list)

        # Display images
        st.sidebar.image(
            uploaded_file, caption="Uploaded Image", use_column_width=False
        )
        st.sidebar.info("Recommended Images:")
        for i in range(5):
            display_image(filenames[indices[0][i]], f"Image {i+1}\n\tPrice=10$")

    else:
        st.error("Error occurred in file upload")
else:
    st.sidebar.info("Upload an image to get recommendations.")
