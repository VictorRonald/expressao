import time
import torch
import numpy as np
from PIL import Image
from advancedliveportrait.nodes import ExpressionEditor  

import torch
print(torch.cuda.is_available())  # Deve imprimir False

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Deve imprimir "cpu"
# tensor = torch.randn(1).to(get_device())
# print(f"Tensor está em: {tensor.device}")  # Deve imprimir 'cpu'

def load_image(image_path):
    """Carrega uma imagem e a converte para tensor PyTorch."""
    pil_image = Image.open(image_path).convert("RGB")
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)  

def process_expression(src_image):
    """Aplica a edição de expressão facial na imagem."""
    params = {
        "rotate_pitch": 0.0,  # "min": -20,  "max": 20
        "rotate_yaw": 0.0,    # "min": -20,  "max": 20
        "rotate_roll": 0.0,   # "min": -20,  "max": 20
        "blink": 0.0,         # "min": -20,  "max": 5
        "eyebrow": 0.0,       # "min": -10,  "max": 15
        "wink": 0.0,          # "min":  0,   "max": 25
        "pupil_x": 0.0,       # "min": -15,  "max": 15
        "pupil_y": 0.0,       # "min": -15,  "max": 15
        "aaa": 0.0,          # "min": -30,  "max": 120
        "eee": 0.0,           # "min": -20,  "max": 15
        "woo": 0.0,           # "min": -20,  "max": 15
        "smile": 1.3,         # "min": -0.3, "max": 1.3
        "src_ratio": 1.0,
        "sample_ratio": 1.0,
        "sample_parts": "OnlyExpression",  # "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
        "crop_factor": 1.7,
        "src_image": src_image,
        "sample_image": None,
        "motion_link": None,
        "add_exp": None
    }

    expression_editor = ExpressionEditor()


    start_time = time.time()
    output = expression_editor.run(**params)
    end_time = time.time()


    output_image = Image.fromarray(np.array(output['result'][0] * 255, dtype=np.uint8)[0])
    
    inference_time = end_time - start_time
    print(f"Tempo de inferência: {inference_time:.4f}s")

    return output_image


if __name__ == "__main__":
    image_path = "data/input/ComfyUI_00118_.png"
    output_path = "data/output/oi.png"

    src_image = load_image(image_path)
    output_image = process_expression(src_image)

    output_image.save(output_path)
    print(f"Imagem salva em {output_path}")
