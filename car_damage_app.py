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

    # Run YOLOv8 inference
    results = model(image_np)

    # Visualize the results
