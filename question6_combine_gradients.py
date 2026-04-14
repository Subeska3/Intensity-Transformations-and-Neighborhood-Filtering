from PIL import Image
import numpy as np

# Load the gradient images
grad_x = Image.open('runway_dog_grad_x.png')
grad_y = Image.open('runway_dog_grad_y.png')

# Ensure they are the same size
if grad_x.size != grad_y.size:
    print("Images have different sizes, resizing...")
    grad_y = grad_y.resize(grad_x.size)

# Create a new image with double width
width, height = grad_x.size
combined = Image.new('RGB', (width * 2, height))

# Paste the images side by side
combined.paste(grad_x.convert('RGB'), (0, 0))
combined.paste(grad_y.convert('RGB'), (width, 0))

# Add labels
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(combined)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Horizontal Gradient (Gx)", fill="white", font=font)
draw.text((width + 10, 10), "Vertical Gradient (Gy)", fill="white", font=font)

# Save the combined image
combined.save('runway_dog_gradients_combined.png')
print('Saved runway_dog_gradients_combined.png')