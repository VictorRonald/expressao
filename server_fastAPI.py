from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO

app = FastAPI()

@app.post("/process")
async def return_same_image(image: UploadFile = File(...)):
    """Apenas retorna a mesma imagem recebida, sem nenhum processamento."""
    try:
        contents = await image.read()
        return StreamingResponse(BytesIO(contents), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
