import subprocess
from PIL import Image, ImageDraw, ImageFont
import os

# Run the test
result = subprocess.run(['python3', 'test.py'], capture_output=True, text=True, cwd='/home/izcy/Desktop/UGM/Smt VII/NLP')
output = result.stdout

# Create image
width = 800
lines = output.strip().split('\n')
line_height = 25
height = len(lines) * line_height + 40

img = Image.new('RGB', (width, height), color='#1e1e1e')
draw = ImageDraw.Draw(img)

# Try to use a monospace font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
except:
    font = ImageFont.load_default()

# Draw text
y = 20
for line in lines:
    # Color code: green for checkmarks, white for others
    if '✓' in line or 'passed' in line.lower():
        color = '#4ec9b0'
    elif 'Testing' in line:
        color = '#dcdcaa'
    else:
        color = '#d4d4d4'

    draw.text((20, y), line, fill=color, font=font)
    y += line_height

# Save
img.save('/home/izcy/Desktop/UGM/Smt VII/NLP/test_screenshot.png')
print("Screenshot saved to test_screenshot.png")
