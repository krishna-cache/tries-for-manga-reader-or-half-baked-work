import torch
from PIL import Image
from transformers import (AutoProcessor,BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration)

mid = "Qwen/Qwen2.5-VL-7B-Instruct"
sq = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     mid,
     torch_dtype=torch.bfloat16,
     quantization_config=sq,
     device_map="auto",
 )

processor = AutoProcessor.from_pretrained(mid)
prompt= """Rules for characters:
- First list every distinct character visible on the page, with a short description of their appearance.
- Do NOT create a new character unless they look clearly different.
- each bubble's "speaker" must be one of the labels in "characters", or "unknown" if you can't tell.
- two bubbles that sit close together in the same panel are usually spoken by the same character, unless the speech tails point to different people.
- always check the speech tail and which character the speech tail is referencing to
- usually if there's no speech tail and the bubble is above a character then it's said by the Character it's above"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image":"01.png",
            },
            {"type": "text", "text": "just read the manga page and assign each bubble to a character by checking the speech tail , Also remove or ignore reading the voice effect texts like 'boom' and other similar voice effects,and return only a list containing tuples of [character ,bubble/text , x coordinate , y coordinate] , no extra texts instead of the list. "+ prompt },
        ],
    }
]


text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True, 
)
image = Image.open("01.png").convert("RGB")
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = inputs.to(model.device)

with torch.no_grad(): # we aint trainin the models so we turning off the notebooks in backgroud 
    output_numbers = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
    )

prompt_length = inputs["input_ids"].shape[1]
new_numbers = output_numbers[:, prompt_length:]

ans = processor.batch_decode(new_numbers, skip_special_tokens=True)[0]
print(ans)

import json
import re


clean = re.sub(r"^```(?:python|json)?\s*|\s*```$", "", ans.strip())

data = json.loads(clean)   
print(data)