from PIL import Image, ImageDraw, ImageFont
import os

out = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(out, exist_ok=True)
path = os.path.join(out, 'sample_image.png')
img = Image.new('RGB', (128, 128), color=(73, 109, 137))
d = ImageDraw.Draw(img)
d.text((10, 50), "Sample", fill=(255,255,0))
img.save(path)
print('Wrote sample image to', path)
