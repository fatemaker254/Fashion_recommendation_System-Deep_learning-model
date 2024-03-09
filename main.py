import tempfile
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

# Set page config first
st.set_page_config(layout="wide")

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Initialize ResNet model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])


def creds_entered():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if (
        st.session_state["user"].strip() == "admin"
        and st.session_state["passwd"].strip() == "admin"
    ):
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter a password")
        elif not st.session_state["user"]:
            st.warning("Please enter a username")
        else:
            st.error("The username or password you entered is incorrect")


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label="Username", value="", key="user", on_change=creds_entered)
        st.text_input(
            label="Password",
            value="",
            key="passwd",
            type="password",
            on_change=creds_entered,
        )
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            st.text_input(
                label="Username", value="", key="user", on_change=creds_entered
            )
            st.text_input(
                label="Password",
                value="",
                key="passwd",
                type="password",
                on_change=creds_entered,
            )
            return False


if authenticate_user():
    st.sidebar.title("Fashion Recommender ")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"]
    )

    def save_uploaded_file(uploaded_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.seek(0)
                return tmp_file.name
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result

    def recommend(features, feature_list):
        neighbors = NearestNeighbors(
            n_neighbors=6, algorithm="brute", metric="euclidean"
        )
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices

    # Display images without resizing
    def display_image(image_path, caption, button_key):
        img = Image.open(image_path).convert("RGB")
        st.image(img, caption=caption, width=150)
        add_to_cart = st.button(f"Add to Cart: {button_key}")
        if add_to_cart:
            st.session_state.cart_items.append((image_path, caption))
        st.write("")  # Add a space below the image

    # Main content
    if "cart_items" not in st.session_state:
        st.session_state.cart_items = []

    if uploaded_file is not None:
        tmp_file_path = save_uploaded_file(uploaded_file)
        if tmp_file_path:
            # feature extract
            features = feature_extraction(tmp_file_path, model)
            if features is not None:
                # recommendation
                indices = recommend(features, feature_list)

                # Display images with Add to Cart button
                st.sidebar.image(
                    uploaded_file, caption="Uploaded Image", use_column_width=False
                )
                st.sidebar.info("Recommended Images:")
                for i in range(5):
                    display_image(
                        filenames[indices[0][i]], f"Image {i+1}\n\tPrice=10$", i + 1
                    )
            else:
                st.error("Error occurred in file processing")
        else:
            st.error("Error occurred in file upload")
    else:
        st.sidebar.info("Upload an image to get recommendations.")

    # Cart page
    if st.sidebar.button("Open Cart"):
        st.sidebar.info("Your Cart:")
        total_price = 0
        for item in st.session_state.cart_items:
            st.sidebar.image(item[0], caption=item[1], width=150)
            total_price += 10  # Assuming each item costs $10
        st.sidebar.info(f"Total Price: ${total_price}")

    place_order_clicked = st.sidebar.button("Place Order")
    if place_order_clicked:
        if st.session_state.cart_items == []:
            st.sidebar.info("Your Cart is Empty")
        else:
            st.session_state.cart_items = []
            st.sidebar.info(
                "Placed Order Successfully!\n\tThank You for Shopping with Us"
            )

    # Adjust layout
    st.sidebar.write("")  # Add space
