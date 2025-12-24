import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm #shows progress bar   

from sklearn.datasets import fetch_openml
#mnist is a dataset of 70 000 handwritten digits (0-9)

mnist = fetch_openml(name='mnist_784', version=1)

print(mnist.keys())

data = mnist["data"]
labels = mnist["target"]


#picking random datapoint to know image and its shape
n = np.random.choice(np.arange(data.shape[0]+1))

print(n)
#n = 252247 

test_img = data.iloc[n].values
test_lable = mnist.target.iloc[n]

print(test_img.shape) #(784,)

side_length = int(np.sqrt(test_img.shape))
reshaped_test_img = test_img.reshape(side_length, side_length)


print("Image label:", test_lable) #9

plt.imshow(reshaped_test_img, cmap='gray')
plt.axis('off')
plt.show()
#the shape will be (28, 28) because sqrt(784) = 28, so the image is 28 pixels by 28 pixels/matrix



