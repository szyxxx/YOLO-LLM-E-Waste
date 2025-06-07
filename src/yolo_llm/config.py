import os

# Directory paths
TEST_IMAGES_DIR = "test_data/test/images"
TEST_LABELS_DIR = "test_data/test/labels"
YOLO_MODEL_PATH = "models/v32.pt"
RESULTS_DIR = "benchmark_results"
YOLO_ANNOTATED_DIR = os.path.join(RESULTS_DIR, "yolo_annotated")
GEMINI_ANNOTATED_DIR = os.path.join(RESULTS_DIR, "gemini_annotated")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Create necessary directories
for directory in [RESULTS_DIR, YOLO_ANNOTATED_DIR, GEMINI_ANNOTATED_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Gemini API configuration
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your Gemini API key

# Class definitions
GEMINI_CLASSES = [
    'Battery', 'Body Weight Scale', 'Calculator', 'Clock', 'DVD Player', 'DVD ROM',
    'Electronic Socket', 'Fan', 'Flashlight', 'Fridge', 'GPU', 'Handphone',
    'Harddisk', 'Insect Killer', 'Iron', 'Keyboard', 'Lamp', 'Laptop',
    'Laptop Charger', 'Microphone', 'Microwave', 'Monitor', 'Motherboard',
    'Mouse', 'PC Case', 'Power Supply', 'Powerbank', 'Printer', 'Printer Ink',
    'Radio', 'Rice Cooker', 'Router', 'Solar Panel', 'Speaker', 'Television',
    'Toaster', 'Walkie Talkie', 'Washing Machine'
] 