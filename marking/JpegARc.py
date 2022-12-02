import random

from PIL import Image
import os
#Path to folder with your generated images
#folder_dir =
#Example
#folder_dir = 'C:\Blobs\Blobs_color\second_set\original_img'

#Path to folder, when you want save processed color images
#folder_s =
#Example
#folder_s = 'C:\Blobs\Blobs_color\second_set\_blur_img'
for filename in os.listdir(folder_dir):
    im = Image.open(os.path.join(folder_dir, filename))
    IMAGE_10 = os.path.join(folder_s, filename)
    rnd = random.randint(1,5)
    im.save(IMAGE_10,"JPEG", quality=rnd*10)





#im10 = Image.open(IMAGE_10)
