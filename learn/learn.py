from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename,askdirectory
import cv2
import numpy as np



img = Image.open("one.png")

#resizing img for ml
img_resized = img.resize((128, 128))

img_array = np.array(img_resized)

img_normalized = img_array / 255.0

print(img_normalized.shape)




