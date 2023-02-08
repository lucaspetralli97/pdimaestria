import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- Cargo Imagen ------------------------------------------
plt.close('all')
I = cv2.imread(f"img02.png") 
# I = cv2.imread(f"img06.png") 
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
imshow(I)

# --- Paso a escalas de grises ------------------------------
Ig = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
imshow(Ig)

# --- Binarizo ---------------------------------------------
th, Ibw = cv2.threshold(Ig, 114, 255, cv2.THRESH_BINARY)    
imshow(Ibw)

# --- Elementos conectados ---------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw, connectivity, cv2.CV_32S)  
imshow(labels)

# *** Observo las áreas de todos los objetos **********
areas = [st[cv2.CC_STAT_AREA] for st in stats]
areas_sorted = sorted(areas)
print(areas_sorted)
for ii, vv in enumerate(areas_sorted):
    print(f"{ii:3d}): {vv:8d}")
# *****************************************************

# --- Filtro por area ---------------------------------------------------------------
Ibw_filtArea = Ibw.copy()
for jj in range(1,num_labels):
    if (stats[jj, cv2.CC_STAT_AREA]>100) or (stats[jj, cv2.CC_STAT_AREA]<30):
        Ibw_filtArea[labels==jj] = 0
imshow(Ibw_filtArea)

# --- Filtro por relacion de aspecto ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw_filtArea, connectivity, cv2.CV_32S)  
imshow(labels)

Ibw_filtAspect = Ibw_filtArea.copy()
for jj in range(1,num_labels):
    rel_asp = stats[jj, cv2.CC_STAT_HEIGHT] / stats[jj, cv2.CC_STAT_WIDTH]
    print(f"{jj:3d}) {rel_asp:5.2f}")
    if (rel_asp<1.5) or (rel_asp >3.0):
        Ibw_filtAspect[labels==jj] = 0
imshow(Ibw_filtAspect)


# --- Resultado parcial ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ibw_filtAspect, connectivity, cv2.CV_32S)  
imshow(labels)
Ipatente = cv2.merge((Ibw_filtAspect, Ibw_filtAspect, Ibw_filtAspect))
for ii in range(1,num_labels):
    cv2.rectangle(Ipatente, tuple(stats[ii,0:2]), tuple(stats[ii,0:2]+stats[ii,2:4]), (255,0,0), 1)
    cv2.putText(Ipatente, f"{ii}", tuple(centroids[ii].astype(int)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.9, color=(255, 0, 0), thickness=1)
imshow(Ipatente)    

# *** Analizo las relaciones de aspecto de todos los objetos *****************************
for ii in range(1,num_labels):
    rel_asp = stats[ii, cv2.CC_STAT_HEIGHT] / stats[ii, cv2.CC_STAT_WIDTH]
    print(f"{ii:3d}) {rel_asp:5.2f}")
 # ***************************************************************************************


# --- Corroboro cercania de otro caracter ----------------------------
DIST_TH = 20
Ipatente_cercania = Ibw_filtAspect.copy()
for ii in range(1, num_labels):
    ch = centroids[ii,:]       # Centroide del caracter
    # --- Obtengo los centroides de los demás caracteres ---------------------------
    objs = np.delete(centroids.copy(), ii, axis=0)  # Elimino centroide actual
    objs = np.delete(objs, 0, axis=0)               # Elimino centroide del fondo
    # --- Calculo distancias -------------------------------------------------------
    aux = objs - ch
    dists = np.sqrt(aux[:,0]**2 + aux[:,1]**2)
    if not any(dists < DIST_TH):
        # print(f"{ii} --> Eliminado")
        Ipatente_cercania[labels==ii] = 0
imshow(Ipatente_cercania)

# --- Resultado final ------------------------------------------------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Ipatente_cercania, connectivity, cv2.CV_32S)  
Ifinal = cv2.merge((Ipatente_cercania, Ipatente_cercania, Ipatente_cercania))
for ii in range(1,num_labels):
    cv2.rectangle(Ifinal, tuple(stats[ii,0:2]), tuple(stats[ii,0:2]+stats[ii,2:4]), (255,0,0), 1)
imshow(Ifinal)    
