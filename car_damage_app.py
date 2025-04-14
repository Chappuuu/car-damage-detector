import streamlit as st
st.set_page_config(page_title="Car Damage Detection", layout="centered")  # ✅ Must be FIRST Streamlit command

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    # Load the YOLOv8 model (pre-trained on COCO dataset)
    return YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for a lightweight model

model = load_model()

def estimate_cost(classes):
    # Define base costs for different types of damage
    base_costs = {"scratch": (100, 200), "dent": (200, 400), "crack": (300, 500)}
    min_total, max_total = 0, 0
    for c in classes:
        cost = base_costs.get(c.lower(), (150, 300))  # Default cost if label not found
        min_total += cost[0]
        max_total += cost[1]
    return min_total, max_total

def get_nearby_garages():
    # Mock data for nearby garages
    return [
        {"name": "AutoFix Garage", "address": "Main Street", "rating": 4.5},
        {"name": "Speedy Repairs", "address": "Broadway", "rating": 4.2},
        {"name": "MaxAuto Repair", "address": "East Avenue", "rating": 4.7},
    ]

# App UI
st.title("🚗 Car Damage Detection & Repair Estimator")

# File uploader for car image
uploaded_file = st.file_uploader("Upload a damaged car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Debug: Check the image shape
    st.write("Image Shape:", image_np.shape)

    # Run YOLOv8 inference
    try:
        results = model(image_np)

        # Debug: Print the YOLOv8 results
        st.write("YOLOv8 Results:", results)

        # Visualize the results
        annotated_frame = results[0].plot()  # Annotated image with detections
        st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

        # Extract detected labels
        detected_labels = []
        if results[0].boxes is not None and results[0].boxes.data is not None:
            # Iterate through the detected boxes
            for box in results[0].boxes.data:
                # Extract the class index and map it to the class name
                class_index = int(box[5])  # Assuming the 6th element is the class index
                class_name = results[0].names[class_index]
                detected_labels.append(class_name)

        st.subheader("🔍 Detected Labels")
        if detected_labels:
            st.write(", ".join(detected_labels))

            # Estimate repair costs
            min_cost, max_cost = estimate_cost(detected_labels)
            st.subheader("💰 Estimated Repair Cost")
            st.write(f"Estimated Cost: **${min_cost} - ${max_cost}**")

            # Display nearby garages
            st.subheader("🧭 Recommended Nearby Garages")
            for g in get_nearby_garages():
                st.markdown(f"**{g['name']}**")
                st.write(f"📍 {g['address']} — ⭐ {g['rating']}")
        else:
            st.info("No damage-related labels detected.")
    except Exception as e:
        st.error(f"An error occurred during inference: {e}")