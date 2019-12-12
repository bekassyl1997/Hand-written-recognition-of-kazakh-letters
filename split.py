from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])  

im = Image.open("sample_demo1_version0.jpg")
p = 61.5
x = 10
y = 10
n = 18
m = 13
images = []
ss = ['ә', 'қ', 'ң', 'ғ', 'ө', 'ұ', 'ү', 'і', 'һ']
for j in range (0,n):
    s = 'C:\\Users\\adilet.baisyn\\Desktop\\big data\\'+ss[j/2]+'\\';
    for i in range (0,m):
        area = (y+i*p,x + j*p,y+i*p+50,x+p*j + 50)
        c = im.crop(area)
        images.append(c)
        c.save(s+str(j)+"_"+str(i)+".jpg", "JPEG", quality=100, optimize=True, progressive=True)