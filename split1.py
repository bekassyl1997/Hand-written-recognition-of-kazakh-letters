import glob
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])  
def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:     
    image = image.resize((28,28), Image.ANTIALIAS)
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return rgb2gray(im_arr)

train_x = np.zeros((12*8,28,28))
train_y = np.zeros((12*8,1))
k = 0
images=glob.glob("*.png")
st = 'C:\\Users\\adilet.baisyn\\Desktop\\big data\\kz_letters_set\\1\\s'
for image in images:
    s = st+str(k+1)+'.png'
    train_x[k]=jpg_image_to_array(s).reshape((28,28))
    k = k+1
print(train_x.shape)