from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM,AutoProcessor
import torch

def read_image(path):
    with open(path, "rb") as f:
        image = Image.open(f).convert("L").convert("RGB")
        return np.array(image)

model = AutoModelForCausalLM.from_pretrained("ragavsachdeva/magiv3", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
processor = AutoProcessor.from_pretrained("ragavsachdeva/magiv3", trust_remote_code=True)

images = [read_image("merged_horizontal.png")]

a= model.predict_detections_and_associations(images, processor)
b= model.predict_ocr(images, processor)
for i in range(len(images)):
    model.visualise_single_image_prediction(images[i], a[i], filename=f"image_{i}.png")
    model.generate_transcript_for_single_image(a[i], b[i], filename=f"transcript_{i}.txt")