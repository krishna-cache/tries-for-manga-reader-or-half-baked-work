import easyocr

reader = easyocr.Reader(["en"], gpu=True)
raw = reader.readtext("z.png", paragraph=True)  

bubbles = []
for box, text in raw:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    bubbles.append({
        "text": text,
        "x": int(min(xs)), "y": int(min(ys))
    })

bubbles.sort(key=lambda b: (b["y"], -b["x"]))

for i, b in enumerate(bubbles, 1):
    print(i, b)
