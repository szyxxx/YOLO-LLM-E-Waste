import google.generativeai as genai
from PIL import Image
from .config import GEMINI_API_KEY, GEMINI_CLASSES

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

def validate_detection(image_crop: Image.Image, yolo_class: str) -> str:
    """
    Validate YOLO detection using Gemini LLM.
    
    Args:
        image_crop: Cropped image of the detected object
        yolo_class: Class predicted by YOLO
        
    Returns:
        str: Validated class name or original YOLO class if validation fails
    """
    prompt = f"""
You are an expert at identifying electronic devices from images.
The current classification is: {yolo_class}
Your task:
- Carefully examine the provided image crop.
- If the current classification is correct, return ONLY the class name from this list: {GEMINI_CLASSES}
- If the current classification is NOT correct, return the correct class name from this list.
- If you are NOT confident about the classification or if the object is NOT in the list, return \"ABSTAIN\"
- Return only the class name or \"ABSTAIN\", nothing else.
"""
    response = gemini_model.generate_content([prompt, image_crop])
    corrected_class = response.text.strip().splitlines()[0]
    
    if corrected_class == "ABSTAIN":
        return yolo_class  # fallback to YOLO if Gemini abstains
    if corrected_class in GEMINI_CLASSES:
        return corrected_class
    return yolo_class 