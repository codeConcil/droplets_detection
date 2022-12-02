
#Not used in latest version

import os


folder_dir = 'C:\Blobs\original_img'
folder_i = 'C:\Blobs\images'
folder_t = 'C:\Blobs\labels'
i=0
for filename in os.listdir(folder_dir):
    i+=1
    print(os.path.splitext(filename)[0])
print(i)