from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import psutil
import time
from advancedliveportrait.nodes import ExpressionEditor

app = FastAPI()

# Carrega o modelo uma vez
expression_editor = ExpressionEditor()

def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{tag}] Memória usada: {mem_mb:.2f} MB")

def load_image(image_file):
    """Carrega uma imagem do UploadFile."""
    pil_image = Image.open(image_file.file).convert("RGB")
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)

def process_expression_v2(src_image, params):
    """Processamento com os novos parâmetros da main.py"""
    try:
        log_memory_usage("Antes do processamento")
        start_time = time.time()

        with torch.no_grad():
            output = expression_editor.run(
                src_image=src_image,
                src_ratio=params.get("src_ratio", 1.0),
                sample_ratio=params.get("sample_ratio", 1.0),
                rotate_pitch=params.get("rotate_pitch", 0.0),
                rotate_yaw=params.get("rotate_yaw", 0.0),
                rotate_roll=params.get("rotate_roll", 0.0),
                blink=params.get("blink", 0.0),
                eyebrow=params.get("eyebrow", 0.0),
                wink=params.get("wink", 0.0),
                pupil_x=params.get("pupil_x", 0.0),
                pupil_y=params.get("pupil_y", 0.0),
                aaa=params.get("aaa", 0.0),
                eee=params.get("eee", 0.0),
                woo=params.get("woo", 0.0),
                smile=params.get("smile", 0.0),
                sample_parts=params.get("sample_parts", "OnlyExpression"),
                crop_factor=params.get("crop_factor", 1.7),
                sample_image=None,
                motion_link=None,
                add_exp=None
            )

        result_image = output['result'][0]
        output_image = Image.fromarray(np.array(result_image * 255, dtype=np.uint8)[0])
        inference_time = time.time() - start_time
        print(f"Tempo de inferência: {inference_time:.4f}s")

        log_memory_usage("Após o processamento")
        del output, result_image
        torch.cuda.empty_cache()

        return output_image

    except Exception as e:
        print(f"Erro no processamento: {str(e)}")
        raise

@app.post("/process")
async def process_image(
    image: UploadFile = File(...),
    emotion: str = Form(None),
    intensity: int = Form(None),
    rotate_pitch: float = Form(0.0),
    rotate_yaw: float = Form(0.0),
    rotate_roll: float = Form(0.0),
    blink: float = Form(0.0),
    eyebrow: float = Form(0.0),
    wink: float = Form(0.0),
    pupil_x: float = Form(0.0),
    pupil_y: float = Form(0.0),
    aaa: float = Form(0.0),
    eee: float = Form(0.0),
    woo: float = Form(0.0),
    smile: float = Form(0.0),
    sample_parts: str = Form("OnlyExpression"),
    crop_factor: float = Form(1.7)
):
    try:
        src_image = load_image(image)

        if emotion and intensity is not None:
            print(f"Modo compatibilidade - Emoção: {emotion}, Intensidade: {intensity}")
            expressions = {
                "feliz": lambda i: {"smile": 0.3 * i},
                "raiva": lambda i: {"eyebrow": -2.0 * i, "smile": -0.1 * i},
                "triste": lambda i: {"eyebrow": 0.2 * i, "smile": -0.3 * i},
                "surpreso": lambda i: {"eyebrow": 5.0 * i, "aaa": 4.0 * i},
            }

            if emotion not in expressions:
                raise ValueError(f"Emoção '{emotion}' não reconhecida.")
            if intensity < 0 or intensity > 10:
                raise ValueError("Intensidade deve estar entre 0 e 10.")

            params = expressions[emotion](intensity)
            params.update({"sample_parts": "OnlyExpression", "crop_factor": 1.7})
        else:
            params = {
                "rotate_pitch": rotate_pitch,
                "rotate_yaw": rotate_yaw,
                "rotate_roll": rotate_roll,
                "blink": blink,
                "eyebrow": eyebrow,
                "wink": wink,
                "pupil_x": pupil_x,
                "pupil_y": pupil_y,
                "aaa": aaa,
                "eee": eee,
                "woo": woo,
                "smile": smile,
                "sample_parts": sample_parts,
                "crop_factor": crop_factor
            }

        output_image = process_expression_v2(src_image, params)

        img_io = BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)
        del output_image, src_image
        return StreamingResponse(img_io, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
