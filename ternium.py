import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os

# Establece la ruta del ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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

os.chdir('imgs')
f_list = os.listdir()

# --- Cargo Imagen ------------------------------------------
plt.close('all')
I = cv2.imread(f_list[2]) 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I)

h, w = I.shape[:2]

# Cambiar el tamaño de la imagen a la mitad
img_resized = cv2.resize(I, (1200,800 ))

# Mostrar la imagen redimensionada
imshow(img_resized)



#--- Paso a escalas de grises ------------------------------
Ig = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
imshow(Ig)

# --- Binarizo ---------------------------------------------
th, Ibw = cv2.threshold(Ig, 120, 255, cv2.THRESH_BINARY)    
# imshow(Ibw)

# --- Invierto ---------------------------------------------
inverted_image = cv2.bitwise_not(Ibw)
# imshow(inverted_image)



#--- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity, cv2.CV_32S)  
imshow(labels)

#----- Reviso cada

# La posición x del borde superior izquierdo del componente conectado (cv2.CC_STAT_LEFT)
# La posición y del borde superior izquierdo del componente conectado (cv2.CC_STAT_TOP)
# El ancho del componente conectado (cv2.CC_STAT_WIDTH)
# La altura del componente conectado (cv2.CC_STAT_HEIGHT)
# El área del componente conectado (cv2.CC_STAT_AREA)

print('----------Código----------')


stats[150,:]
stats[152,:]
stats[154,:]
stats[216,:]

print('----------Barras----------')

stats[218,:]
stats[219,:]
stats[221,:]

# --- Filtro por altura, ancho y area ---------------------------------------------------------------
Ibw_filtalt = inverted_image.copy()
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_HEIGHT]<60) or (stats[jj, cv2.CC_STAT_HEIGHT]>90):
        Ibw_filtalt[labels ==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_WIDTH]<5) or (stats[jj, cv2.CC_STAT_WIDTH]>40):
        Ibw_filtalt[labels==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_AREA]<500) or (stats[jj, cv2.CC_STAT_AREA]>1500):
        Ibw_filtalt[labels==jj] = 0


imshow(Ibw_filtalt)

# ------- Aplica filtro gaussiano --------------
blurred = cv2.GaussianBlur(Ibw_filtalt, (25, 25), 0)
imshow(blurred)

#--- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blurred, connectivity, cv2.CV_32S)  
imshow(labels)


#----------- Creo una máscara utilizando los datos de stats para el código que ahora es 1 solo componente conectado -------


# Obtén los valores de la caja delimitadora del componente conectado específico
component_id = 154 # hay que encontrar la manera de que sea el unico componente
left = stats[component_id, cv2.CC_STAT_LEFT]
top = stats[component_id, cv2.CC_STAT_TOP]
width = stats[component_id, cv2.CC_STAT_WIDTH]
height = stats[component_id, cv2.CC_STAT_HEIGHT]

# Crea una imagen en blanco de la misma forma que la imagen original
mask = np.zeros_like(img_resized, dtype=np.uint8)

# Asigna un valor de píxel de 1 a los píxeles dentro de la caja delimitadora del componente conectado
mask[top:top+int(round(height*1.01, 0)), left:left+int(round(width*1.01,0))] = 255 # sumamos un margen para no cortar caracteres

imshow(mask)

img_resized.dtype
mask.dtype

img_resized.shape
mask.shape

# Definir el rango de valores para la máscara
lower = 0
upper = 255

# Aplicar cv2.inRange() para obtener una máscara con valores de 0 y 255
mask = cv2.inRange(mask, lower, upper)

# --- Invierto ---------------------------------------------
inverted_mask = cv2.bitwise_not(mask)
imshow(inverted_mask)


# Aplica la máscara ROI a la imagen original para obtener la ROI correspondiente
roi = cv2.bitwise_and(img_resized, img_resized, mask=inverted_mask)

imshow(roi)


# ------------ Corto la máscara --------------------


# Encuentra los contornos de la máscara
contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encuentra el contorno más grande
largest_contour = max(contours, key=cv2.contourArea)

# Obtén el rectángulo que encierra el contorno
x, y, w, h = cv2.boundingRect(largest_contour)

# Recorta la imagen
cropped_img = roi[y:y+h, x:x+w]

cropped_img.shape

imshow(cropped_img)

# Convierte la imagen recortada a escala de grises
cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
# imshow(cropped_gray)

# Aplicar umbralización
ret, thresh = cv2.threshold(cropped_gray, 127, 255, cv2.THRESH_BINARY)

# Crear kernel rectangular
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# Aplicar erosión
erosion = cv2.erode(thresh, kernel, iterations = 1)
# imshow(erosion)

# --- Invierto ---------------------------------------------
inverted_image2 = cv2.bitwise_not(erosion)
imshow(inverted_image2)



# Encuentra los bordes de la imagen recortada
edges = cv2.Canny(cropped_gray, 50, 150)

# Encuentra los esquinas en la imagen de bordes
corners = cv2.goodFeaturesToTrack(edges, 100, 0.01, 10)

# Calcula el ángulo medio de inclinación de las esquinas
angle = 0
if corners is not None:
    for corner in corners:
        x, y = corner[0]
        angle += np.arctan2(y - h / 2, x - w / 2)

angle /= len(corners)

# Rota la imagen para enderezar el texto
rows, cols = cropped_img.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle * 180 / np.pi, 1)
rotated = cv2.warpAffine(cropped_img, M, (cols, rows))

imshow(rotated)


text = pytesseract.image_to_string(cropped_img)
# text = pytesseract.image_to_string(I)
print(text)
tokens = text.split()
print(tokens)

for token in tokens:
    if len(token) == 13:
        print(token.upper())

