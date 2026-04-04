from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import shutil, uuid, os
from pipeline import fashion_pipeline

app = FastAPI(title="Fashion AI API")

# ── CORS — add your Vercel URL here ──
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://your-site.vercel.app"          # ← replace with your Vercel URL
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "Fashion AI API is running"}

@app.post("/recommend")
async def recommend(
    image   : UploadFile = File(...),
    occasion: str        = Form(...)
):
    # Save uploaded image to temp file
    temp_path = f"/tmp/{uuid.uuid4().hex}.jpg"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        # Run heavy ML pipeline in thread so FastAPI doesn't block
        result = await run_in_threadpool(fashion_pipeline, temp_path, occasion)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)   # always cleanup temp file

    return result