from transformers.pytorch_utils import prune_conv1d_layer
from transformers import AutoModel,AutoModelForCausalLM,AutoProcessor
import numpy as np
from PIL import Image
import torch
import os


from PIL import Image

# Open the images
img1 = Image.open('01.png')
img2 = Image.open('02.png')

# 1. For Horizontal Merge (Side-by-Side)
total_width = img1.width + img2.width
max_height = max(img1.height, img2.height)

# Create a blank canvas
new_img_horizontal = Image.new('RGB', (total_width, max_height))

# Paste the images
new_img_horizontal.paste(img2, (0, 0))
new_img_horizontal.paste(img1, (img2.width, 0))
new_img_horizontal.save('merged_horizontal.png')




images = [
        "merged_horizontal.png"
    ]

def read_image_as_np_array(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

images = [read_image_as_np_array(image) for image in images]

model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
m2= AutoModelForCausalLM.from_pretrained("ragavsachdeva/magiv3", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
processor = AutoProcessor.from_pretrained("ragavsachdeva/magiv3", trust_remote_code=True)
with torch.no_grad():
    results = m2.predict_detections_and_associations(images,processor)
    text_bboxes_for_all_images = [x["texts"] for x in results]
    ocr_results = m2.predict_ocr(images,processor)

for i in range(len(images)):
    model.visualise_single_image_prediction(images[i], results[i], filename=f"image_{i}.png")
    model.generate_transcript_for_single_image(results[i], ocr_results[i], filename=f"transcript_{i}.txt")

print(ocr_results)
print("-----------------------")
print(results)
print("-----------------------")
print(text_bboxes_for_all_images)

print(len(images))
