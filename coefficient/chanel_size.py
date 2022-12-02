import cv2
from PIL import Image


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"
# Read the original image
#Path to original image, which you using for calculate coefficient
img = cv2.imread('sample.jpg')

# Display original image

# cv2.imshow('Original', img)

# cv2.waitKey(0)

# Convert to graycsale

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = 255
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('Canny Edge Detection', img)
cv2.imwrite('white.png', img)
cv2.waitKey(0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection

img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Canny Edge Detection

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection

# Display Canny Edge Detection Image
im = Image.fromarray(edges)

im = im.convert('HSV')
pix = im.load()
buffer = []
isStart = True
counter = 1
buffer = 0
summ = 0
lines = 5
coeff = 500

#Resolution video or image, which you going send to detection
target_height = 1080

mult = target_height / im.size[1]

for i in range(int(im.size[0] / 4 * 3), im.size[0]):
    for j in range(0, im.size[1]):
        color = pix[i, j]
        if color[2] > 100 and counter <=lines:
            pix[i, j] = (0, 255, 255)
            k = j+1
            buffer+=1
            while pix[i,k][2] < 100:

                color1 = (0, 255, 255)
                pix[i,k] = color1
                buffer+=1
                k+=1
            pix[i,k] = (0, 255, 255)
            buffer+= 1
            summ += buffer * mult
            buffer = 0
            counter +=1
            break

summ = summ / lines
pix_size = coeff / summ
print(pix_size)
pix_size = pix_size / 1000
print(pix_size)
print(toFixed(pix_size, 2))

with open('coef.txt', 'w') as f:
    f.write(toFixed(pix_size, 6))


im.show()
im = im.convert("RGBA")
im.save("kenny.png")
#cv2.imshow('Canny Edge Detection', edges)

cv2.waitKey(0)

cv2.destroyAllWindows()
