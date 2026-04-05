from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import shutil, uuid, os
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion AI API")

# ── CORS — add your Vercel URL here ──
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,https://renclo.vercel.app"
).split(",")

logger.info(f"✅ CORS Allowed Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """Health check endpoint"""
    logger.info("✅ Health check requested")
    return {"status": "Fashion AI API is running", "version": "1.0"}

@app.post("/recommend")
async def recommend(
    image   : UploadFile = File(...),
    occasion: str        = Form(...)
):
    """
    Main API endpoint for fashion recommendations
    
    Args:
        image: Image file uploaded by user
        occasion: Occasion type (casual, formal, party, etc.)
    """
    
    # Validate inputs
    if not image:
        logger.error("❌ No image provided")
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not occasion or occasion.strip() == "":
        logger.error("❌ No occasion provided")
        raise HTTPException(status_code=400, detail="Occasion is required")
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if image.content_type not in allowed_types:
        logger.error(f"❌ Invalid file type: {image.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (JPG, PNG, WebP). Got: {image.content_type}"
        )
    
    # Create cross-platform temp file
    temp_dir = tempfile.gettempdir()
    file_ext = Path(image.filename).suffix or ".jpg"
    temp_filename = f"{uuid.uuid4().hex}{file_ext}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    logger.info(f"📥 Processing image: {image.filename}")
    
    try:
        # Save uploaded file
        try:
            content = await image.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            file_size = os.path.getsize(temp_path)
            if file_size > 10 * 1024 * 1024:
                logger.error(f"❌ File too large: {file_size} bytes")
                raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")
            
            logger.info(f"✅ File saved successfully ({file_size} bytes)")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to save file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to save file: {str(e)}")
        
        # Run heavy ML pipeline in thread so FastAPI doesn't block
        try:
            logger.info(f"🔄 Running pipeline with occasion: {occasion.lower().strip()}")
            result = await run_in_threadpool(fashion_pipeline, temp_path, occasion.lower().strip())
            
            if result.get("success"):
                logger.info("✅ Pipeline completed successfully")
            else:
                logger.error(f"❌ Pipeline failed: {result.get('error')}")
            
            return result
            
        except Exception as pipeline_error:
            logger.error(f"❌ Pipeline error: {str(pipeline_error)}", exc_info=True)
            return {
                "success": False,
                "error": f"Pipeline error: {str(pipeline_error)}",
                "outfits": []
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
        
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🧹 Cleaned up temp file")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_error}")