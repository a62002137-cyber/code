from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sqlite3, jwt, bcrypt, os, cv2, numpy as np, io, tempfile, logging
from PIL import Image, ImageFilter, ImageEnhance
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Style Transfer API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer()
SECRET_KEY = "style-transfer-key-2024"
ALGORITHM = "HS256"
DATABASE_PATH = "app.db"
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

STYLE_PRESETS = {
    "pixar": {"saturation": 1.5, "contrast": 1.3, "brightness": 1.1, "blur": 1},
    "anime": {"saturation": 1.8, "contrast": 1.4, "brightness": 1.0, "blur": 0},
    "realis": {"saturation": 0.9, "contrast": 1.1, "brightness": 0.95, "blur": 0},
    "lego": {"saturation": 2.0, "contrast": 1.5, "brightness": 1.2, "blur": 2},
    "fiksi_ilmiah": {"saturation": 0.8, "contrast": 1.6, "brightness": 0.8, "blur": 0},
    "kartun_retro": {"saturation": 1.6, "contrast": 1.2, "brightness": 1.15, "blur": 1}
}

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()

def init_database():
    with get_db_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cursor = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if not cursor.fetchone():
            password_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", password_hash))
        conn.commit()

def create_token(username: str):
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=24)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError: raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError: raise HTTPException(status_code=401, detail="Invalid token")

def apply_style_transfer(image_data: bytes, preset: str) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB': image = image.convert('RGB')
        params = STYLE_PRESETS.get(preset, STYLE_PRESETS["realis"])
        
        if params["saturation"] != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(params["saturation"])
        if params["contrast"] != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(params["contrast"])
        if params["brightness"] != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(params["brightness"])
        if params["blur"] > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=params["blur"]))
        
        if preset == "pixar":
            image_array = np.array(image)
            image_array = cv2.bilateralFilter(image_array, 15, 50, 50)
            image = Image.fromarray(image_array)
        elif preset == "anime":
            image_array = np.array(image)
            image_array = cv2.bilateralFilter(image_array, 9, 200, 200)
            image = Image.fromarray(image_array)
        elif preset == "lego":
            width, height = image.size
            image = image.resize((width//8, height//8), Image.NEAREST)
            image = image.resize((width, height), Image.NEAREST)
        elif preset == "fiksi_ilmiah":
            image_array = np.array(image)
            image_array[:, :, 0] = image_array[:, :, 0] * 0.7
            image_array[:, :, 2] = np.clip(image_array[:, :, 2] * 1.3, 0, 255)
            image = Image.fromarray(image_array.astype(np.uint8))
        elif preset == "kartun_retro":
            image_array = np.array(image)
            image_array = (image_array // 32) * 32
            image = Image.fromarray(image_array)
        
        output_buffer = io.BytesIO()
        image.save(output_buffer, format='JPEG', quality=85)
        return output_buffer.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

def process_video_frame(video_data: bytes, preset: str) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_data)
            temp_path = temp_file.name
        try:
            cap = cv2.VideoCapture(temp_path)
            ret, frame = cap.read()
            cap.release()
            if not ret: raise HTTPException(status_code=400, detail="Could not extract frame from video")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            frame_buffer = io.BytesIO()
            image.save(frame_buffer, format='JPEG')
            frame_bytes = frame_buffer.getvalue()
            return apply_style_transfer(frame_bytes, preset)
        finally: os.unlink(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

init_database()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/login")
async def login(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")
    if not username or not password: raise HTTPException(status_code=400, detail="Username and password required")
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(username)
        return {"access_token": token, "token_type": "bearer"}

@app.post("/process")
@limiter.limit("5/minute")
async def process_file(request: Request, file: UploadFile = File(...), preset: str = Form(...), username: str = Depends(verify_token)):
    if preset not in STYLE_PRESETS: raise HTTPException(status_code=400, detail="Invalid preset")
    file_data = await file.read()
    file_type = file.content_type
    try:
        if file_type.startswith('image/'): result_data = apply_style_transfer(file_data, preset)
        elif file_type.startswith('video/'): result_data = process_video_frame(file_data, preset)
        else: raise HTTPException(status_code=400, detail="Unsupported file type")
        return Response(content=result_data, media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename=styled_{file.filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
