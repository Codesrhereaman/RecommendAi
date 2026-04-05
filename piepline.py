# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import cv2
import torch
import mediapipe as mp
import google.generativeai as genai
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import KMeans
from PIL import Image
import os
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. LOAD MODEL — downloads from URL on startup
#    Store best_model.pth on Google Drive / S3
#    and paste the direct download link below
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_PATH = "best_model.pth"

def load_segmentation_model():
    """Load model with comprehensive error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Model file not found at {MODEL_PATH}. "
                f"Please ensure model/best_model.pth exists."
            )
        
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info("✅ Segmentation model loaded successfully")
        return model
        
    except FileNotFoundError as e:
        logger.error(f"❌ {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}", exc_info=True)
        raise

model_predict = load_segmentation_model()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# 2. SEGMENTATION
# ─────────────────────────────────────────────
def predict(image_path):
    """Segment person from image with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            original_image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")
        
        if original_image.size == 0:
            raise ValueError("Image is empty or has zero dimensions")
        
        logger.info(f"Image loaded: {original_image.shape}")
        
        inp = transform(image=original_image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model_predict(inp)
            mask   = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        return binary_mask, original_image
        
    except Exception as e:
        logger.error(f"❌ Segmentation error: {str(e)}", exc_info=True)
        raise

# ─────────────────────────────────────────────
# 3. BODY SHAPE
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose

def get_body_shape(original_image, binary_mask):
    """Detect body shape with comprehensive error handling"""
    try:
        h, w = original_image.shape[:2]
        mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        person_only  = cv2.bitwise_and(original_image, original_image, mask=mask_resized)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(cv2.cvtColor(person_only, cv2.COLOR_RGB2BGR))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def px(lmk): return np.array([lmk.x * w, lmk.y * h])

            shoulder_width = np.linalg.norm(px(lm[11]) - px(lm[12]))
            hip_width      = np.linalg.norm(px(lm[23]) - px(lm[24]))
            mid_left       = (px(lm[11]) + px(lm[23])) / 2
            mid_right      = (px(lm[12]) + px(lm[24])) / 2
            waist_width    = np.linalg.norm(mid_left - mid_right) * 0.80
            method         = "MediaPipe"
        else:
            logger.warning("⚠️ MediaPipe failed → fallback to row-slicing")
            ys, _ = np.where(mask_resized > 0)
            
            if len(ys) == 0:
                logger.warning("❌ Empty mask - cannot determine body shape")
                return "Rectangle"
            
            top, bottom = ys.min(), ys.max()
            height = bottom - top
            
            if height == 0:
                logger.warning("❌ Zero height - cannot determine body shape")
                return "Rectangle"
            
            def row_width(y):
                if y < 0 or y >= mask_resized.shape[0]:
                    return 0
                xs = np.where(mask_resized[y] > 0)[0]
                return int(xs.max() - xs.min()) if len(xs) > 0 else 0
            
            shoulder_width = row_width(top + int(0.25 * height))
            waist_width    = row_width(top + int(0.50 * height))
            hip_width      = row_width(top + int(0.75 * height))
            method         = "Mask Row-Slicing"

        if shoulder_width == 0 or hip_width == 0 or waist_width == 0:
            logger.warning("❌ Zero widths - defaulting to Rectangle")
            return "Rectangle"

        if abs(shoulder_width - hip_width) < 20 and waist_width < shoulder_width * 0.85:
            shape = "Hourglass"
        elif hip_width > shoulder_width * 1.1:
            shape = "Triangle"
        elif shoulder_width > hip_width * 1.1:
            shape = "Inverted Triangle"
        else:
            shape = "Rectangle"

        logger.info(f"✅ Body Shape: {shape} [{method}]")
        return shape
        
    except Exception as e:
        logger.error(f"❌ Body shape detection error: {str(e)}", exc_info=True)
        return "Rectangle"

# ─────────────────────────────────────────────
# 4. SKIN TONE
# ─────────────────────────────────────────────
def get_skin_tone(original_image, binary_mask):
    """Detect skin tone with comprehensive error handling"""
    try:
        h, w = original_image.shape[:2]
        mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        person_only  = cv2.bitwise_and(original_image, original_image, mask=mask_resized)

        hsv        = cv2.cvtColor(person_only, cv2.COLOR_RGB2HSV)
        skin_mask  = cv2.inRange(hsv,
                        np.array([0, 20, 70],   dtype=np.uint8),
                        np.array([25, 180, 255], dtype=np.uint8))
        skin_pixels = original_image[skin_mask > 0]

        if len(skin_pixels) < 200:
            logger.warning("⚠️ HSV fallback → face region")
            ys, _ = np.where(mask_resized > 0)
            
            if len(ys) == 0:
                logger.warning("❌ No pixels found - using default skin tone")
                return "Medium", "Neutral"
            
            top, bottom  = ys.min(), ys.max()
            face_crop    = original_image[top : top + int(0.20 * (bottom - top)), :]
            skin_pixels  = face_crop.reshape(-1, 3)

        if len(skin_pixels) == 0:
            logger.warning("❌ No skin pixels found - using defaults")
            return "Medium", "Neutral"

        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        kmeans.fit(skin_pixels.astype(float))
        r, g, b    = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        brightness = (r + g + b) / 3

        tone      = "Fair"    if brightness > 200 else "Medium" if brightness > 140 else "Dark"
        undertone = "Warm"    if r > b + 10       else "Cool"   if b > r + 10       else "Neutral"

        logger.info(f"✅ Skin: {tone} {undertone}")
        return tone, undertone
        
    except Exception as e:
        logger.error(f"❌ Skin tone detection error: {str(e)}", exc_info=True)
        return "Medium", "Neutral"

# ─────────────────────────────────────────────
# 5. GEMINI
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured")
else:
    logger.warning("⚠️ GEMINI_API_KEY not set - AI recommendations will not work")

def get_recommendations(body_shape, skin_tone, undertone, occasion):
    """Get fashion recommendations with error handling"""
    if not GEMINI_API_KEY:
        logger.error("❌ Gemini API key missing")
        return "⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable.", []

    role = """You are an expert AI Fashion Stylist with deep knowledge of:
- Body shape dressing rules (hourglass, rectangle, triangle, inverted triangle)
- Color theory and which colors complement different skin tones and undertones
- Occasion-appropriate outfit styling for any event or setting"""

    input_data = f"""
[USER PROFILE]
Body Shape     : {body_shape}
Skin Tone      : {skin_tone}
Skin Undertone : {undertone}
Occasion       : {occasion}
"""
    task = """[YOUR TASK]
Suggest exactly 3 complete outfit combinations for this user.
Rules:
- Every outfit must have: Top, Bottom, Shoes (+ optional Jacket or Accessory)
- Every item must have an exact specific color
- Colors must complement the skin tone and undertone
- Cuts must flatter the body shape
- Outfits must suit the occasion

Output STRICTLY in this format:
Outfit 1: [item+color], [item+color], [item+color], [item+color]
Outfit 2: [item+color], [item+color], [item+color], [item+color]
Outfit 3: [item+color], [item+color], [item+color], [item+color]"""

    prompt = f"{role}\n\n{input_data}\n\n{task}"

    try:
        logger.info("🔄 Calling Gemini API for recommendations...")
        model    = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt, timeout=30)
        raw      = response.text.strip()
        outfits  = []
        for line in raw.split("\n"):
            if line.startswith("Outfit"):
                items = line.split(":", 1)[1].strip().split(",")
                outfits.append([i.strip() for i in items])
        logger.info(f"✅ Generated {len(outfits)} outfit recommendations")
        return raw, outfits
    except Exception as e:
        logger.error(f"❌ Gemini API error: {str(e)}", exc_info=True)
        return f"Error getting recommendations: {str(e)}", []

# ─────────────────────────────────────────────
# 6. MASTER PIPELINE
# ─────────────────────────────────────────────
def fashion_pipeline(image_path: str, occasion: str) -> dict:
    """Complete fashion recommendation pipeline with error handling"""
    try:
        logger.info(f"🎯 Starting pipeline for: {image_path}, occasion: {occasion}")
        
        try:
            binary_mask, original_image = predict(image_path)
        except Exception as e:
            logger.error(f"❌ Image prediction failed: {str(e)}")
            raise
        
        try:
            body_shape = get_body_shape(original_image, binary_mask)
        except Exception as e:
            logger.error(f"❌ Body shape detection failed: {str(e)}")
            body_shape = "Rectangle"
        
        try:
            skin_tone, undertone = get_skin_tone(original_image, binary_mask)
        except Exception as e:
            logger.error(f"❌ Skin tone detection failed: {str(e)}")
            skin_tone, undertone = "Medium", "Neutral"
        
        try:
            raw_text, outfits_list = get_recommendations(body_shape, skin_tone, undertone, occasion)
        except Exception as e:
            logger.error(f"❌ Recommendations failed: {str(e)}")
            raw_text, outfits_list = f"Error: {str(e)}", []

        result = {
            "success"   : True,
            "body_shape": body_shape,
            "skin_tone" : skin_tone,
            "undertone" : undertone,
            "occasion"  : occasion,
            "raw_text"  : raw_text,
            "outfits"   : outfits_list
        }
        
        logger.info("✅ Pipeline completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "outfits": [],
            "body_shape": "Unknown",
            "skin_tone": "Unknown",
            "undertone": "Unknown"
        }