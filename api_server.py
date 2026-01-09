"""
Photo Editor REST API - Use from any web or mobile app

A FastAPI server that exposes all photo editing features via HTTP endpoints.
Perfect for integrating with your web/mobile apps!

Usage:
    # Start the server
    python api_server.py

    # Or with custom port
    python api_server.py --port 8080

    # Then call from anywhere:
    curl -X POST "http://localhost:8000/upscale" -F "file=@image.png" -o upscaled.png
    curl -X POST "http://localhost:8000/remove-bg" -F "file=@photo.jpg" -o nobg.png
    curl -X POST "http://localhost:8000/convert/banner" -F "file=@art.png" -o banner.png

Endpoints:
    POST /upscale          - AI upscale image (Real-ESRGAN)
    POST /remove-bg        - Remove background (rembg)
    POST /vectorize        - Convert to SVG
    POST /resize           - Smart resize
    POST /convert/{preset} - Convert to preset format (banner, postcard, etc.)
    POST /pipeline         - Full processing pipeline
    GET  /presets          - List available presets
    GET  /health           - Health check
"""

import io
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from photo_editor import (
    PhotoEditor, UpscaleModel, VectorMode, ResizeMode
)
from format_converter import FormatConverter
from presets import get_preset, ALL_PRESETS, list_presets


# Global editor instance (reuse for efficiency)
editor: Optional[PhotoEditor] = None
converter: Optional[FormatConverter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global editor, converter
    editor = PhotoEditor()
    converter = FormatConverter(output_base=tempfile.mkdtemp())
    print("Photo Editor API ready!")
    yield
    # Cleanup on shutdown
    if converter:
        shutil.rmtree(converter.output_base, ignore_errors=True)


app = FastAPI(
    title="SuperCharged Photo Editor API",
    description="AI-powered image processing for web and mobile apps",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    features: List[str]


class PresetInfo(BaseModel):
    name: str
    width: int
    height: int
    description: str
    category: str


class ProcessingResponse(BaseModel):
    success: bool
    message: str
    original_size: Optional[List[int]] = None
    output_size: Optional[List[int]] = None


# ============================================================================
# Utility Functions
# ============================================================================

async def save_upload_to_temp(file: UploadFile) -> Path:
    """Save uploaded file to temp directory."""
    suffix = Path(file.filename).suffix if file.filename else ".png"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def create_image_response(image_path: Path, filename: str = "output.png") -> StreamingResponse:
    """Create a streaming response for an image file."""
    def iterfile():
        with open(image_path, "rb") as f:
            yield from f

    media_type = "image/png"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    elif filename.endswith(".svg"):
        media_type = "image/svg+xml"

    return StreamingResponse(
        iterfile(),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        features=[
            "ai_upscale",
            "remove_background",
            "vectorize",
            "resize",
            "format_convert",
            "pipeline"
        ]
    )


@app.get("/presets")
async def get_presets():
    """List all available format presets."""
    presets = []
    for name, preset in ALL_PRESETS.items():
        category = "social" if "twitter" in name or "instagram" in name or "facebook" in name or "linkedin" in name or "youtube" in name else \
                   "print" if "postcard" in name or "flyer" in name or "poster" in name or "business" in name else \
                   "web"
        presets.append({
            "name": name,
            "width": preset.width,
            "height": preset.height,
            "description": preset.description,
            "category": category
        })
    return {"presets": presets}


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Query(4, ge=2, le=4, description="Upscale factor (2 or 4)"),
    model: str = Query("general", description="Model: general, anime, or fast")
):
    """
    AI upscale an image using Real-ESRGAN.

    - **file**: Image file to upscale
    - **scale**: 2 or 4 (default: 4)
    - **model**: general, anime, or fast (default: general)

    Returns the upscaled image.
    """
    temp_input = await save_upload_to_temp(file)

    try:
        # Map model name to enum
        model_map = {
            "general": UpscaleModel.GENERAL_X4 if scale == 4 else UpscaleModel.GENERAL_X2,
            "anime": UpscaleModel.ANIME_X4,
            "fast": UpscaleModel.FAST_X4,
        }
        upscale_model = model_map.get(model, UpscaleModel.GENERAL_X4)

        # Create output path
        output_path = temp_input.parent / f"{temp_input.stem}_upscaled{temp_input.suffix}"

        # Upscale
        result = editor.ai_upscale(temp_input, output_path, scale=scale, model=upscale_model)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        # Return the image
        response = create_image_response(
            Path(result.output_path),
            f"{Path(file.filename).stem}_upscaled_{scale}x.png"
        )

        return response

    finally:
        # Cleanup temp files
        temp_input.unlink(missing_ok=True)


@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    alpha_matting: bool = Query(False, description="Enable alpha matting for better edges")
):
    """
    Remove background from an image using AI.

    - **file**: Image file
    - **alpha_matting**: Enable for better edge detection (slower)

    Returns PNG with transparent background.
    """
    temp_input = await save_upload_to_temp(file)

    try:
        output_path = temp_input.parent / f"{temp_input.stem}_nobg.png"

        result = editor.remove_background(temp_input, output_path, alpha_matting=alpha_matting)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        return create_image_response(
            Path(result.output_path),
            f"{Path(file.filename).stem}_nobg.png"
        )

    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/vectorize")
async def vectorize_image(
    file: UploadFile = File(...),
    mode: str = Query("illustration", description="Mode: photo, illustration, logo, pixel_art")
):
    """
    Convert raster image to vector SVG.

    - **file**: Image file
    - **mode**: photo, illustration, logo, or pixel_art

    Returns SVG file.
    """
    temp_input = await save_upload_to_temp(file)

    try:
        mode_map = {
            "photo": VectorMode.PHOTO,
            "illustration": VectorMode.ILLUSTRATION,
            "logo": VectorMode.LOGO,
            "pixel_art": VectorMode.PIXEL_ART,
        }
        vector_mode = mode_map.get(mode, VectorMode.ILLUSTRATION)

        output_path = temp_input.parent / f"{temp_input.stem}.svg"

        result = editor.vectorize(temp_input, output_path, mode=vector_mode)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        return create_image_response(
            Path(result.output_path),
            f"{Path(file.filename).stem}.svg"
        )

    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: Optional[int] = Query(None, description="Target width"),
    height: Optional[int] = Query(None, description="Target height"),
    scale: Optional[float] = Query(None, description="Scale factor (e.g., 0.5, 2.0)"),
    mode: str = Query("lanczos", description="Mode: lanczos, bicubic, bilinear, nearest")
):
    """
    Resize an image.

    Provide either width/height OR scale factor.

    - **file**: Image file
    - **width**: Target width in pixels
    - **height**: Target height in pixels
    - **scale**: Scale factor (alternative to width/height)
    - **mode**: Resize algorithm
    """
    if width is None and height is None and scale is None:
        raise HTTPException(
            status_code=400,
            detail="Provide width, height, or scale parameter"
        )

    temp_input = await save_upload_to_temp(file)

    try:
        mode_map = {
            "lanczos": ResizeMode.LANCZOS,
            "bicubic": ResizeMode.BICUBIC,
            "bilinear": ResizeMode.BILINEAR,
            "nearest": ResizeMode.NEAREST,
        }
        resize_mode = mode_map.get(mode, ResizeMode.LANCZOS)

        suffix = f"_{width}x{height}" if width and height else f"_{scale}x" if scale else "_resized"
        output_path = temp_input.parent / f"{temp_input.stem}{suffix}{temp_input.suffix}"

        result = editor.smart_resize(
            temp_input, output_path,
            width=width, height=height, scale=scale,
            mode=resize_mode
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        return create_image_response(
            Path(result.output_path),
            f"{Path(file.filename).stem}_resized.png"
        )

    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/convert/{preset_name}")
async def convert_to_preset(
    preset_name: str,
    file: UploadFile = File(...)
):
    """
    Convert image to a preset format.

    - **preset_name**: Preset name (e.g., banner, postcard, icon, flyer)
    - **file**: Image file

    Use GET /presets to see all available presets.
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset: {preset_name}. Use GET /presets to see available options."
        )

    temp_input = await save_upload_to_temp(file)

    try:
        output_path = temp_input.parent / f"{temp_input.stem}{preset.suffix}.png"

        result = converter.convert(temp_input, preset_name, output_path)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        return create_image_response(
            Path(result.output_path),
            f"{Path(file.filename).stem}{preset.suffix}.png"
        )

    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    remove_bg: bool = Query(True, description="Remove background"),
    upscale: bool = Query(False, description="AI upscale"),
    upscale_model: str = Query("general", description="Upscale model"),
    vectorize: bool = Query(False, description="Create SVG"),
    resize_width: Optional[int] = Query(None, description="Resize width"),
    resize_height: Optional[int] = Query(None, description="Resize height")
):
    """
    Run the full processing pipeline.

    Returns a ZIP file with all processed outputs.
    """
    import zipfile

    temp_input = await save_upload_to_temp(file)
    output_dir = Path(tempfile.mkdtemp())

    try:
        # Map upscale model
        model_map = {
            "general": UpscaleModel.GENERAL_X4,
            "anime": UpscaleModel.ANIME_X4,
            "fast": UpscaleModel.FAST_X4,
        }

        resize_config = None
        if resize_width or resize_height:
            resize_config = {"width": resize_width, "height": resize_height}

        results = editor.process_full_pipeline(
            temp_input,
            output_dir,
            remove_bg=remove_bg,
            ai_upscale=upscale,
            upscale_model=model_map.get(upscale_model, UpscaleModel.GENERAL_X4),
            create_vector=vectorize,
            resize_config=resize_config
        )

        # Create ZIP with all outputs
        zip_path = output_dir / "processed.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                if result.success and result.output_path:
                    zipf.write(result.output_path, Path(result.output_path).name)

        # Return ZIP
        def iterfile():
            with open(zip_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={Path(file.filename).stem}_processed.zip"
            }
        )

    finally:
        temp_input.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)


@app.post("/batch/convert")
async def batch_convert(
    files: List[UploadFile] = File(...),
    preset_name: str = Query(..., description="Preset to convert to")
):
    """
    Batch convert multiple images to a preset format.

    Returns a ZIP file with all converted images.
    """
    import zipfile

    preset = get_preset(preset_name)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset: {preset_name}"
        )

    temp_dir = Path(tempfile.mkdtemp())

    try:
        output_paths = []

        for file in files:
            temp_input = temp_dir / file.filename
            content = await file.read()
            temp_input.write_bytes(content)

            output_path = temp_dir / f"{temp_input.stem}{preset.suffix}.png"
            result = converter.convert(temp_input, preset_name, output_path)

            if result.success:
                output_paths.append(Path(result.output_path))

        # Create ZIP
        zip_path = temp_dir / "batch_converted.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for path in output_paths:
                zipf.write(path, path.name)

        def iterfile():
            with open(zip_path, "rb") as f:
                yield from f

        return StreamingResponse(
            iterfile(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=batch_{preset_name}.zip"
            }
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Photo Editor API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        SuperCharged Photo Editor API Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /upscale          - AI upscale (Real-ESRGAN)         ║
║    POST /remove-bg        - Remove background                ║
║    POST /vectorize        - Convert to SVG                   ║
║    POST /resize           - Smart resize                     ║
║    POST /convert/{{preset}} - Convert to format               ║
║    POST /pipeline         - Full processing pipeline         ║
║    GET  /presets          - List all presets                 ║
║    GET  /health           - Health check                     ║
╠══════════════════════════════════════════════════════════════╣
║  Docs: http://{args.host}:{args.port}/docs                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
