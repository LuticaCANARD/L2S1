#!/usr/bin/env python3
"""Serve the locally trained image classifier on loopback for the visual demo."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from transformers import SiglipImageProcessor, SiglipVisionModel


def create_app(directory):
    report = json.loads((directory / 'report.json').read_text())
    checkpoint = directory / 'classifier.joblib'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report['classifier_sha256']:
        raise ValueError('Classifier checkpoint differs from evaluated checkpoint')
    classifier = joblib.load(checkpoint)  # Only a locally produced, hash-checked file.
    processor = SiglipImageProcessor.from_pretrained(report['model'], revision=report['revision'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SiglipVisionModel.from_pretrained(report['model'], revision=report['revision'], torch_dtype=torch.float32).to(device).eval()
    torch.set_num_threads(4)
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'ready': True, 'model': report['model'], 'classifier': report['classifier'], 'checkpoint': report['classifier_sha256'], 'device': device}

    @app.post('/classify')
    async def classify(request: Request):
        if request.headers.get('content-type', '').split(';')[0] not in {'image/jpeg', 'image/png', 'image/webp'}:
            raise HTTPException(415, 'Choose JPEG, PNG or WebP')
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > 8 * 1024 * 1024:
                raise HTTPException(413, 'Image exceeds 8 MiB')
        try:
            image = Image.open(io.BytesIO(content))
            if image.width * image.height > 24_000_000:
                raise HTTPException(413, 'Image exceeds 24 megapixels')
            image = image.convert('RGB')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, 'Image cannot be decoded')
        started = time.perf_counter()
        with torch.inference_mode():
            inputs = processor(images=[image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)], return_tensors='pt').to(device)
            pooled = model(**inputs).pooler_output.float().mean(dim=0, keepdim=True)
            features = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
        scores = classifier.predict_proba(features)[0]
        winner = int(np.argmax(scores))
        return {'selected': report['classes'][winner], 'scores': [{'id': label, 'option_probability': float(score)} for label, score in zip(report['classes'], scores)], 'latency_ms': (time.perf_counter() - started) * 1000, 'image_sha256': hashlib.sha256(content).hexdigest(), 'classifier_sha256': report['classifier_sha256'], 'policy': 'forced top-1', 'device': device}

    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--port', type=int, default=8766)
    args = p.parse_args()
    uvicorn.run(create_app(args.model_dir), host='127.0.0.1', port=args.port)
