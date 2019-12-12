import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

#def rgb2gray(rgb):
#   return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])  
def resize_RGBA_RGB():
    st = 'C:\\Users\\adilet.baisyn\\Desktop\\big data\\DATA_SET\\'
    # ne zabud pro h
    for i in range(1,43):
        images = glob.glob(st+'l'+str(i)+'\\'+'*.png')
        j = 0
        for image in images:
            s = st+'l'+str(i)+'\\'+'s'+str(j+1)+'.png'
            png = Image.open(s)
            if(png.size != (28,28)):
                png = png.resize((28,28), Image.ANTIALIAS)
                png.save(s,'png', quality=80)
            if(png.mode == 'RGBA'):
                png = png.resize((28,28), Image.ANTIALIAS)
                png.load()
                background = Image.new("RGB", png.size, (255, 255, 255))
                background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
                background.save(s, 'png', quality=80)
            
            j = j+1
def rgb2gray():
    st = 'C:\\Users\\adilet.baisyn\\Desktop\\big data\\DATA_SET\\'
    # ne zabud pro h
    for i in range(1,43):
        images = glob.glob(st+'l'+str(i)+'\\'+'*.png')
        j = 0
        for image in images:
            s = st+'l'+str(i)+'\\'+'s'+str(j+1)+'.png'
            image = Image.open(s)
            image = image.convert(mode='L')
            image.save(s,'png', quality=80)
            j = j+1
rgb2gray()