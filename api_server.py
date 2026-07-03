"""
Photo Editor REST API - AI-powered image processing over HTTP.

Usage:
    python api_server.py             # Start on port 8000
    python api_server.py --port 8080

    curl -X POST "http://localhost:8000/upscale" -F "file=@image.png" -o upscaled.png
    curl -X POST "http://localhost:8000/remove-bg" -F "file=@photo.jpg" -o nobg.png
    curl -X POST "http://localhost:8000/convert/banner" -F "file=@art.png" -o banner.png

Endpoints:
    POST /upscale          - AI upscale image (Real-ESRGAN)
    POST /remove-bg        - Remove background (rembg)
    POST /vectorize        - Convert to SVG
    POST /resize           - Smart resize
    POST /convert/{preset} - Convert to preset format (banner, postcard, ...)
    POST /pipeline         - Full processing pipeline
    POST /batch/convert    - Batch convert to one preset
    GET  /presets          - List available presets
    GET  /health           - Health check
"""

import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from format_converter import FormatConverter
from photo_editor import PhotoEditor, ResizeMode, UpscaleModel, VectorMode
from presets import ALL_PRESETS, get_preset

VERSION = "1.0.0"

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".zip": "application/zip",
}

VECTOR_MODES = {mode.value: mode for mode in VectorMode}
RESIZE_MODES = {mode.value: mode for mode in ResizeMode}

editor: Optional[PhotoEditor] = None
converter: Optional[FormatConverter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared editor/converter instances for the server's lifetime."""
    global editor, converter
    editor = PhotoEditor()
    converter = FormatConverter(output_base=tempfile.mkdtemp())
    print("Photo Editor API ready!")
    yield
    shutil.rmtree(converter.output_base, ignore_errors=True)


app = FastAPI(
    title="Photo Editor API",
    description="AI-powered image processing for web and mobile apps",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    features: List[str]


# ============================================================================
# Helpers
# ============================================================================

async def save_upload_to_temp(file: UploadFile) -> Path:
    """Persist an uploaded file to a temp path for processing."""
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        return Path(temp_file.name)


def file_response(path: Path, filename: str) -> StreamingResponse:
    """Stream a file back as an attachment with the right media type."""
    def iterfile():
        with open(path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type=MEDIA_TYPES.get(Path(filename).suffix.lower(), "image/png"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def require_success(result) -> Path:
    """Return the output path of a processing result, or raise a 500."""
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    return Path(result.output_path)


def zip_files(paths: List[Path], zip_path: Path) -> Path:
    """Bundle files into a ZIP archive."""
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in paths:
            archive.write(path, path.name)
    return zip_path


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=VERSION,
        features=["ai_upscale", "remove_background", "vectorize",
                  "resize", "format_convert", "pipeline"],
    )


@app.get("/presets")
async def get_presets():
    """List all available format presets."""
    return {
        "presets": [
            {
                "name": name,
                "width": preset.width,
                "height": preset.height,
                "description": preset.description,
                "category": preset.output_folder.split("/")[0],
            }
            for name, preset in ALL_PRESETS.items()
        ]
    }


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Query(4, ge=2, le=4, description="Upscale factor (2 or 4)"),
    model: str = Query("general", description="Model: general, anime, or fast"),
):
    """AI upscale an image using Real-ESRGAN."""
    temp_input = await save_upload_to_temp(file)
    try:
        result = editor.ai_upscale(
            temp_input,
            temp_input.parent / f"{temp_input.stem}_upscaled{temp_input.suffix}",
            scale=scale, model=UpscaleModel.from_name(model, scale),
        )
        return file_response(require_success(result),
                             f"{Path(file.filename).stem}_upscaled_{scale}x.png")
    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    alpha_matting: bool = Query(False, description="Better edge detection (slower)"),
):
    """Remove the background from an image. Returns PNG with transparency."""
    temp_input = await save_upload_to_temp(file)
    try:
        result = editor.remove_background(
            temp_input, temp_input.parent / f"{temp_input.stem}_nobg.png",
            alpha_matting=alpha_matting,
        )
        return file_response(require_success(result),
                             f"{Path(file.filename).stem}_nobg.png")
    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/vectorize")
async def vectorize_image(
    file: UploadFile = File(...),
    mode: str = Query("illustration", description="Mode: photo, illustration, logo, pixel_art"),
):
    """Convert a raster image to vector SVG."""
    temp_input = await save_upload_to_temp(file)
    try:
        result = editor.vectorize(
            temp_input, temp_input.parent / f"{temp_input.stem}.svg",
            mode=VECTOR_MODES.get(mode, VectorMode.ILLUSTRATION),
        )
        return file_response(require_success(result),
                             f"{Path(file.filename).stem}.svg")
    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: Optional[int] = Query(None, description="Target width"),
    height: Optional[int] = Query(None, description="Target height"),
    scale: Optional[float] = Query(None, description="Scale factor (e.g., 0.5, 2.0)"),
    mode: str = Query("lanczos", description="Mode: lanczos, bicubic, bilinear, nearest"),
):
    """Resize an image by width/height or scale factor."""
    if width is None and height is None and scale is None:
        raise HTTPException(status_code=400,
                            detail="Provide width, height, or scale parameter")

    temp_input = await save_upload_to_temp(file)
    try:
        suffix = f"_{width}x{height}" if width and height else f"_{scale}x" if scale else "_resized"
        result = editor.smart_resize(
            temp_input, temp_input.parent / f"{temp_input.stem}{suffix}{temp_input.suffix}",
            width=width, height=height, scale=scale,
            mode=RESIZE_MODES.get(mode, ResizeMode.LANCZOS),
        )
        return file_response(require_success(result),
                             f"{Path(file.filename).stem}_resized.png")
    finally:
        temp_input.unlink(missing_ok=True)


@app.post("/convert/{preset_name}")
async def convert_to_preset(preset_name: str, file: UploadFile = File(...)):
    """Convert an image to a preset format. See GET /presets for options."""
    preset = get_preset(preset_name)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset: {preset_name}. Use GET /presets to see available options.",
        )

    temp_input = await save_upload_to_temp(file)
    try:
        result = converter.convert(
            temp_input, preset_name,
            temp_input.parent / f"{temp_input.stem}{preset.suffix}.png",
        )
        return file_response(require_success(result),
                             f"{Path(file.filename).stem}{preset.suffix}.png")
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
    resize_height: Optional[int] = Query(None, description="Resize height"),
):
    """Run the full processing pipeline. Returns a ZIP of all outputs."""
    temp_input = await save_upload_to_temp(file)
    output_dir = Path(tempfile.mkdtemp())
    try:
        resize_config = None
        if resize_width or resize_height:
            resize_config = {"width": resize_width, "height": resize_height}

        results = editor.process_full_pipeline(
            temp_input, output_dir,
            remove_bg=remove_bg,
            ai_upscale=upscale,
            upscale_model=UpscaleModel.from_name(upscale_model),
            create_vector=vectorize,
            resize_config=resize_config,
        )
        outputs = [Path(r.output_path) for r in results if r.success and r.output_path]
        zip_path = zip_files(outputs, output_dir / "processed.zip")
        return file_response(zip_path, f"{Path(file.filename).stem}_processed.zip")
    finally:
        temp_input.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)


@app.post("/batch/convert")
async def batch_convert(
    files: List[UploadFile] = File(...),
    preset_name: str = Query(..., description="Preset to convert to"),
):
    """Batch convert multiple images to one preset. Returns a ZIP."""
    preset = get_preset(preset_name)
    if preset is None:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_name}")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        outputs = []
        for file in files:
            temp_input = temp_dir / file.filename
            temp_input.write_bytes(await file.read())

            result = converter.convert(
                temp_input, preset_name,
                temp_dir / f"{temp_input.stem}{preset.suffix}.png",
            )
            if result.success:
                outputs.append(Path(result.output_path))

        zip_path = zip_files(outputs, temp_dir / "batch_converted.zip")
        return file_response(zip_path, f"batch_{preset_name}.zip")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Photo Editor API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Photo Editor API Server                         ║
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

    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
