import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg


def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:  
    im_arr = np.array(image)
    im_arr = im_arr.reshape((image.size[1], image.size[0]))                                   
  return im_arr

def getData():
    st = 'C:\\Users\\adilet.baisyn\\Desktop\\big data\\DATA_SET\\'
    train_x = []#np.zeros((19913,28,28))
    train_y = []#np.zeros(19913)
    k = 0
    # ne zabud pro h
    for i in range(1,43):
        images = glob.glob(st+'l'+str(i)+'\\'+'*.png')
        j = 0
        for image in images:
            s = st+'l'+str(i)+'\\'+'s'+str(j+1)+'.png'
            train_x.append(jpg_image_to_array(s).reshape((28,28)))
            train_y.append(i)
            k = k+1
            j = j+1
    return np.array(train_x),np.array(train_y)
    
(train_x,train_y) = getData()
#im = Image.fromarray(train_x[-2])
#im.show()
#im = Image.fromarray(train_x[5555])
#im.show()
#im = Image.fromarray(train_x[1])
#im.show()
train_x = train_x.reshape(-1, 28,28)
train_x = train_x.astype('float32')
train_x = train_x / 255

print(train_x)
print(train_y.shape)
train_y = train_y.reshape(-1)
np.save("x.npy", train_x)
np.save("y.npy", train_y)
    #print(train_x.shape)
    #img = Image.fromarray(train_x[-1])
    #img.show()