import cv2
import os

os.chdir('imgs')
f_list = os.listdir()

# Defininimos función para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

# Load the image
I = cv2.imread(f_list[1]) 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I)

# Pre-process the image
gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
imshow(thresh)

# --- Invierto ---------------------------------------------
inverted_image = cv2.bitwise_not(thresh)
imshow(inverted_image)


# Find the region of the image that contains the code
contours = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

for c in contours:
    # Check if the contour is large enough to be the code
    if cv2.contourArea(c) > 1000:
        # Extract the code from the image and resize it to a fixed size
        x, y, w, h = cv2.boundingRect(c)
        code_image = gray[y:y+h, x:x+w]
        code_image = cv2.resize(code_image, (250, 50), interpolation=cv2.INTER_LINEAR)

        # Extract the 13-digit code from the resized image using template matching
        template = cv2.imread('template.jpg', 0)
        res = cv2.matchTemplate(code_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            code_x, code_y = max_loc
            code_w, code_h = template.shape[::-1]
            code = code_image[code_y:code_y+code_h, code_x:code_x+code_w]
            break

# The code is now stored in the code variable
