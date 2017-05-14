# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:10:52 2017

@author: PAUL
"""
import cv2
import matplotlib.pyplot as plt


# methode de lecture d'une image    
def read_image(path):
    img = cv2.imread(path)
    
    return img

# methode de consersion d'une image en niveau de gris
def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    return gray

# methode d'affichage d'histogramme
def show_histo(histo):
    print(len(histo))
    print(max(histo))
    try:
        plt.hist(histo, normed=1, bins=len(histo))
        plt.axis([0, len(histo), 0, max(histo)])
        plt.show()
    except Exception as e:
        print(e)
    
# methode de chargement de la base pour les moments de Hu   
#def chargement_base_image2(base_images):
#    matrice_base_images=[]
#    taille=len(base_images)
#    for i in range(taille):
#        image1 = uts.read_image(base_images[i])
#        arr = uts.to_gray(np.asarray(image1))
#        matrice_base_images.append({'nom':base_images[i],'matrice':arr})
#    
#    return matrice_base_images