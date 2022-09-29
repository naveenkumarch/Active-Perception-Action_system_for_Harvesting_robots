#importting the required libraries
import cv2
import os 
import numpy as np
import zipfile
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast,RandomRotation,RandomCrop,RandomFlip,RandomZoom, Resizing

#Defining the random Augmentations to be used for new samples creation 
transform = tf.keras.Sequential([
    RandomRotation((-0.5,0.5), fill_mode='wrap', interpolation='nearest'),
    RandomFlip("horizontal_and_vertical"),
    RandomZoom((-0.1,-0.5)),
    Resizing(100, 100, interpolation='nearest')
])

os.mkdir('.../working/augmented/')

count = 0
!cd '.../working/augmented/'
path = "./augmented"
# Reading files and creating 250 new samples using the augmentations
for count in range(250):
    for dirname, _, filenames in os.walk("../input/strawberry-texture-pics/Texture_pics/Spoiled_fruits"):
        for filename in filenames:
            #print(filename)
            image = cv2.imread(str("../input/strawberry-texture-pics/Texture_pics/Spoiled_fruits"+'/'+filename))
            image = tf.expand_dims(image, 0)
            # Augmenting the image
            augment = transform(image)
            transformed_image = augment[0]
            img = np.array(transformed_image)
            name = "spoiled_"+str(count)+".PNG"
            cv2.imwrite(os.path.join(path , name), img)



# Creating a ZIP file out of the newly generated files    
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
      
zipf = zipfile.ZipFile('Python.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('./augmented/', zipf)
zipf.close()