import cv2
import numpy as np
np.set_printoptions(suppress=True)
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
I = cv2.imread(f_list[4]) 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I)

#----------- Llevo a un mismo tamaño ----------------

# h, w = I.shape[:2]

# # Cambiar el tamaño de la imagen a la mitad - ¿Qué resize conviene? 
# img_resized = cv2.resize(I, (1200,800 )) #mantener relación de aspecto - todas sacadas en la misma resolución, llevar todo a HD. 

# # Mostrar la imagen redimensionada
# imshow(img_resized)


#--- Paso a escalas de grises ------------------------------
Ig = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
# imshow(Ig)

# --- Binarizo ---------------------------------------------
th, Ibw = cv2.threshold(Ig, 120, 255, cv2.THRESH_BINARY)    
# imshow(Ibw)

# --- Invierto ---------------------------------------------
inverted_image = cv2.bitwise_not(Ibw)
imshow(inverted_image)



#--- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity, cv2.CV_32S)  
imshow(labels)


# La posición x del borde superior izquierdo del componente conectado (cv2.CC_STAT_LEFT)
# La posición y del borde superior izquierdo del componente conectado (cv2.CC_STAT_TOP)
# El ancho del componente conectado (cv2.CC_STAT_WIDTH)
# La altura del componente conectado (cv2.CC_STAT_HEIGHT)
# El área del componente conectado (cv2.CC_STAT_AREA)


# --- Reviso componentes ------------------------------------------

def get_component_stats(labels, stats, labels_of_interest):
    
    idx = labels_of_interest[0]  
    min_width, min_height, min_area = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT], stats[idx, cv2.CC_STAT_AREA]
    max_width, max_height, max_area = min_width, min_height, min_area

    for label in labels_of_interest[1:]:
       
        min_width = min(min_width, stats[label, cv2.CC_STAT_WIDTH])
        min_height = min(min_height, stats[label, cv2.CC_STAT_HEIGHT])
        min_area = min(min_area, stats[label, cv2.CC_STAT_AREA])
        max_width = max(max_width, stats[label, cv2.CC_STAT_WIDTH])
        max_height = max(max_height, stats[label, cv2.CC_STAT_HEIGHT])
        max_area = max(max_area, stats[label, cv2.CC_STAT_AREA])

    return (min_width, min_height, min_area), (max_width, max_height, max_area)


labels_of_interest = [68,66,64,61,57,52,48,45,43,42,41,40,39]

r = get_component_stats(labels, stats, labels_of_interest)



min_width  = r[0][0]
max_withd   = r[1][0]
min_height = r[0][1]
max_height = r[1][1]
min_area   = r[0][2]
max_area   = r[1][2]


#hacerlo relativo al ancho y alto de la imagen 

# --- Filtro por altura, ancho y area ---------------------------------------------------------------
Ibw_filtalt = inverted_image.copy()

for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_WIDTH]<min_width) or (stats[jj, cv2.CC_STAT_WIDTH]>max_withd):
        Ibw_filtalt[labels==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_HEIGHT]<min_height) or (stats[jj, cv2.CC_STAT_HEIGHT]>max_height):
        Ibw_filtalt[labels ==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_AREA]<min_area) or (stats[jj, cv2.CC_STAT_AREA]>max_area):
        Ibw_filtalt[labels==jj] = 0

imshow(Ibw_filtalt)

# --- Filtro por altura, ancho y area ---------------------------------------------------------------
Ibw_filtalt = inverted_image.copy()


h, w = I.shape[:2]

for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_WIDTH]<w * 0.011) or (stats[jj, cv2.CC_STAT_WIDTH]>w * 0.040):
        Ibw_filtalt[labels==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_HEIGHT]<h * 0.060) or (stats[jj, cv2.CC_STAT_HEIGHT]>h * 0.074):
        Ibw_filtalt[labels ==jj] = 0
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_AREA]<(h*w) * 0.00065) or (stats[jj, cv2.CC_STAT_AREA]>(h*w) * 0.0015):
        Ibw_filtalt[labels==jj] = 0


imshow(Ibw_filtalt)

# calculando los rangos para cada filtro como % de la altura y ancho de la imagen para evitar usar el resize

h, w = I.shape[:2]

w * 0.066

min_width / w #1,2% #1,5%
max_withd / w #2,1% #3,8%
min_height / h #6,7% #6,1%
max_height / h #7,3 #6,8%
min_area / (h*w)  #  #0.065
max_area / (h*w) #  #0.1429

(h*w) * 0.00065



# # ------- Aplica filtro gaussiano --------------
# blurred = cv2.GaussianBlur(Ibw_filtalt, (31, 31), 0)
# imshow(blurred)

#--- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw_filtalt, connectivity, cv2.CV_32S)  
imshow(labels)

# ----------- Calculamos la mediana de los centroides para filtar ruido -----------

type(centroids)

# calcular la mediana de cada columna 
column_medians = np.median(centroids, axis=0)

# imprimir los resultados
print("La mediana de la primera columna es:", column_medians[0])
print("La mediana de la segunda columna es:", column_medians[1])



#meter filtro de mayor area para que sólo quede un componente 

#distancia entre centroides 


#----------- Creo una máscara utilizando los datos de stats para el código que ahora es 1 solo componente conectado -------


# Obtén los valores de la caja delimitadora del componente conectado específico
component_id = 2 # hay que encontrar la manera de que sea el unico componente
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
# imshow(inverted_mask)


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
imshow(erosion)

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


# thickness skeleton 


text = pytesseract.image_to_string(Ibw_filtalt)
# text = pytesseract.image_to_string(I)
print(text)
tokens = text.split()
print(tokens)

# for token in tokens:
#     if len(token) == 13:
#         print(token.upper())

#subconjunto de imagenes buenas
#train vs test
#conjunto más heavy + análisis 
#qué se podría mejorar 


