import pytesseract
import re
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#Open image
im = Image.open("A.png")

#Define configuration that only whitelists number characters
custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'

#Find the numbers in the image
numbers_string = pytesseract.image_to_string(im, config=custom_config)

#Remove all non-number characters
numbers_int = re.sub(r'[a-z\n]', '', numbers_string.lower())

#print the output
print(numbers_int)