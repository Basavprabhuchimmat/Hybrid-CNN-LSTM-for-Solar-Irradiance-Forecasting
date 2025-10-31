import io
import os
import json
from typing import Optional

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from .utils import save_upload_file

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save uploaded file and return dummy prediction
    out_path = os.path.join("uploads", file.filename)
    await save_upload_file(file, out_path)
    return JSONResponse({"status": "ok", "filename": file.filename, "prediction": 123.45})


@router.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
