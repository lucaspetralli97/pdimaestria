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
I = cv2.imread(f_list[1]) 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I)

h, w = I.shape[:2]


# Cambiar el tamaño de la imagen a la mitad
img_resized = cv2.resize(I, (1024,768 ))

# Mostrar la imagen redimensionada
imshow(img_resized)



#--- Paso a escalas de grises ------------------------------
Ig = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
imshow(Ig)

# --- Binarizo ---------------------------------------------
th, Ibw = cv2.threshold(Ig, 120, 255, cv2.THRESH_BINARY)    
imshow(Ibw)

# --- Invierto ---------------------------------------------
inverted_image = cv2.bitwise_not(Ibw)
imshow(inverted_image)

#--- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity, cv2.CV_32S)  
imshow(labels)

#----- Reviso cada

# La posición x del borde superior izquierdo del componente conectado.
# La posición y del borde superior izquierdo del componente conectado.
# El ancho del componente conectado.
# La altura del componente conectado.
# El área del componente conectado.

stats[67,:]
stats[66,:]
stats[65,:]

# Podemos pensar en varios filtros: por altura, por ancho, area y distancia entre los centroides 
# todo para obtener la parte donde está el código, encontrar una máscara y simplificar trabajo del OCR

# *** Observo las áreas de todos los objetos **********
# areas = [st[cv2.CC_STAT_AREA] for st in stats]
# areas_sorted = sorted(areas)
# print(areas_sorted)
# for ii, vv in enumerate(areas_sorted):
#     print(f"{ii:3d}): {vv:8d}")

# --- Filtro por altura ---------------------------------------------------------------
Ibw_filtalt = inverted_image.copy()
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_HEIGHT]<60) or (stats[jj, cv2.CC_STAT_HEIGHT]>90):
        Ibw_filtalt[labels==jj] = 0
imshow(Ibw_filtalt)

text = pytesseract.image_to_string(Ibw_filtalt)
# text = pytesseract.image_to_string(I)
print(text)
tokens = text.split()
print(tokens)

for token in tokens:
    if len(token) == 13:
        print(token.upper())

