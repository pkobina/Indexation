#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:06:40 2017

@author: paul
"""
#import cv2
import numpy as np
import random
import os
import math
#import image_slicer

import utils as uts

from os.path import splitext


BIG_NUMBER = math.pow(10, 10)

#extraction du repertoire et extensions des images de notre base
def extraction_repertoire_extension(image_requete):
    filename, extension = splitext(image_requete)
    rep = os.path.dirname(image_requete)
    extraction=[rep, extension]    
    
    return extraction

# lecture des images de la base:repertoire=repertoire de la base d'images, 
# extension=extension des images de la base
def lecture_base_images(repertoire, extension, image_requete):
    base_images=[]
    for file in os.listdir(repertoire):
        if file.endswith(extension):
            if os.path.join(repertoire, file) != image_requete:
                base_images.append(os.path.join(repertoire, file))
                
    return base_images

# methode de chargement de la base pour histogramme et niveau de gris pour moments de Hu
def chargement_base_image(base_images, label):
    matrice_base_images=[]
    taille=len(base_images)
    for i in range(taille):
        image = uts.read_image(base_images[i])
        if label:
            image = uts.to_gray(np.asarray(image))
        matrice_base_images.append({'nom':base_images[i],'matrice':image})
    
    return matrice_base_images

# construction de l'histogramme couleur
def buildHistogram(im, echelle_reduction):
    histo=[]
    k=int(256/echelle_reduction)
    R = [0]*k
    G = [0]*k
    B = [0]*k
    try:
        height, width, channel = im.shape
        for i in range(height):
            for j in range(width):
                R[int(im[i, j, 0]/echelle_reduction)] += 1
                G[int(im[i, j, 1]/echelle_reduction)] += 1
                B[int(im[i, j, 2]/echelle_reduction)] += 1
        # concatenation des histogramme
        histo.extend(R)
        histo.extend(G)
        histo.extend(B)
    except Exception as e:
        print(e)
    return histo

# calcul de la distance de Mahalanobis
def comparaisonHistoMahalanobis(histoA, histoB):
    distance = 0.0
    for i in range(len(histoA)):
        distance += abs(histoA[i] - histoB[i])
    # calcul de la distance    
    newDistance = distance/sum(histoA)
    
    return newDistance

# methode de calcul des histogrammes de la base d'images
def Calcul_histogramme_base(images, echelle_reduction):
    tableau_histogramme = []
    taille = len(images)
    for i in range(taille):
        histo = buildHistogram(images[i]['matrice'], echelle_reduction)
        tableau_histogramme.append({'nom':images[i]['nom'], 'histo': histo})        
    
    return tableau_histogramme

# methode de calcul de distance de a partir des histogramme
def calcul_distance_histogramme_base_images(base_images, image_requete, echelle_reduction):
    tableau_distance_histogramme = []
   
    matrice_image_requete = uts.read_image(image_requete)
    matrice_base_images_couleurs = chargement_base_image(base_images, label = False)
    histogramme_base_images = Calcul_histogramme_base(matrice_base_images_couleurs, echelle_reduction)
    histo_image_requete = buildHistogram(matrice_image_requete, echelle_reduction)
    taille = len(histogramme_base_images)
     
    for i in range(taille):
        distance=comparaisonHistoMahalanobis(histo_image_requete, histogramme_base_images[i]['histo'])
        tableau_distance_histogramme.append({'nom':histogramme_base_images[i]['nom'],'distance':distance})
    tableau_distance_histogramme=sorted( tableau_distance_histogramme, key=lambda dct: dct['distance'])
    
    return tableau_distance_histogramme

# methode de calcul de distance de la base a partir des moments de Hu
def calcul_distance_hu_base_images(base_images_gris, image_requete):
    tableau_distance_hu=[]
    
    matrice_image_requete = uts.read_image(image_requete)
    matrice_base_images_gris = chargement_base_image(base_images_gris, label = True)
    taille=len(matrice_base_images_gris)
    matrice_image_requete_gris = uts.to_gray(matrice_image_requete)
    
    for i in range(taille):
        distance = distance_moments_hu_img(matrice_base_images_gris[i]['matrice'], matrice_image_requete_gris)
        tableau_distance_hu.append({'nom': matrice_base_images_gris[i]['nom'], 'distance':distance})
        
    tableau_distance_hu = sorted(tableau_distance_hu, key = lambda dct: dct['distance'])
    
    return  tableau_distance_hu

###############################################################################
# implementation de l'algorithme de knn
###############################################################################
# methode d'affichage du resultat de knn
def afficher_knn(k, tableau_distance):
    print("\n** Resultat du knn: k = %i **\n" % k)
    print("\t\tImages\t\t\tDistance")
    for i in range(k):
        image = os.path.basename(tableau_distance[i]['nom'])
        print("\t%s\t\t\t%3f"%(image, tableau_distance[i]['distance']))

# methode de knn avec les differents descripteurs (couleurs, moments de Hu et distance globale)
def knn(base_images, image_requete, option):
    k = int(input("Veuillez specifier k: "))
    # generation des resultats a partir des donnees d'apprentissage
    tableau_distance_histogramme=[]
    tableau_distance_hu=[]
    try:
        if option == 0:
                       
            echelle_reduction=int(input("Veuillez specifiez l'échelle de réduction: "))
            tableau_distance_histogramme=calcul_distance_histogramme_base_images(base_images, image_requete,echelle_reduction)
            #affichage des k plus proches voisins
            afficher_knn(k, tableau_distance_histogramme)
        elif option == 1:
            #chargement des images de la base
            tableau_distance_hu=calcul_distance_hu_base_images(base_images, image_requete)
            #affichage des k plus proches voisins
            afficher_knn(k, tableau_distance_hu)
        elif option == 2:
            distance_similarite_globale=[]
            echelle_reduction=int(input("Veuillez specifiez l'échelle de réduction: "))
            try:
                poids=float(input("Veuillez specifiez le poids a appliquer sur la similarite: "))
            except:
                poids=0.5
            tableau_distance_histogramme=calcul_distance_histogramme_base_images(base_images, image_requete, echelle_reduction)
            tableau_distance_hu=calcul_distance_hu_base_images(base_images, image_requete)
            
            taille=len(tableau_distance_hu)
            for i in range(taille):
                distance=similarite_globale(poids, tableau_distance_histogramme[i]['distance'], tableau_distance_hu[i]['distance'])
                distance_similarite_globale.append({'nom':tableau_distance_histogramme[i]['nom'], 'distance':distance })
                
            distance_similarite_globale=sorted(distance_similarite_globale, key=lambda dct: dct['distance'])
            #affichage des k plus proches voisins
            afficher_knn(k, distance_similarite_globale)
        else:
            print("Mauvais choix\n")
            
    except Exception as e:
        print(e)
  
##################################################################################
# implementation de l'algorithme de k-means
##################################################################################
# methode d'initialisation des centroides des clusters
def initialize_centroids(k, base, option):
    centroids=[]
  
    for i in range(k):
        
        c=random.choice(base)
        if option==1:
            centroids.append(c['hu'])
        else:
            centroids.append(c['histo'])
        
    return centroids

# methode de recuperation des elements d'un cluster
def elements_cluster(base, numero_cluster, option):
    elements_clusters=[]
    taille=len(base)
    for i in range(taille):
        if base[i]['groupe']== numero_cluster:
            if option==0:
                elements_clusters.append(base[i]['histo'])
            else:
                elements_clusters.append(base[i]['hu'])
            
    return elements_clusters

# methode de reevaluation des centroides
def recalculate_centroids(k, base, centroids, option):
    temp=[]
    for m in range(k):
        elem=elements_cluster(base,k,option)
        longueur_base=len(elem)
        if longueur_base != 0:
            one_item=len(elem[0])
            centroid=[0]*one_item
            for i in range(one_item):
                for j in range(longueur_base):
                    if option==0:
                        centroid[i]+=base[j]['histo'][i]
                    else:
                        centroid[i]+=base[j]['hu'][i]
                    
                centroid[i]/=one_item
                temp.append(centroid)
            centroids[m]=temp
                     
    return centroids

# methode de mise a jour des clusters
def update_clusters(base, k, centroids, option):
    isStillMoving = 0
    
    taille=len(base)
    
    for i in range(taille):
         bestMinimum = BIG_NUMBER
         currentCluster = 0
         for j in range(k):
             if option==1:
                 distance = distance_moments_hu_moments(base[i]['hu'], centroids[j])
             else:
                 distance=comparaisonHistoMahalanobis(base[i]['histo'], centroids[j])
             #print("distance=", distance)
             if(distance < bestMinimum):
                 bestMinimum = distance
                 currentCluster = j
         base[i]['groupe']=currentCluster
    
         if(base[i]['groupe'] is None or base[i]['groupe'] != currentCluster):
            base[i]['groupe']=currentCluster
            isStillMoving = 1
    
    
    return isStillMoving 

# methode d'amelioration du clustering
def perform_kmeans(base_hu, centroids, k, option):
    isStillMoving = 1
    while(isStillMoving):
        centroids=recalculate_centroids(k, base_hu, centroids, option)
        isStillMoving = update_clusters(base_hu, k, centroids, option)
    
# methode d'affichage de resultat du clustering k-means
def print_results(k, base):
    taille=len(base)
    for i in range(k):
        print("Cluster %i comporte:" % (i+1))
        for j in range(taille):
            if(base[j]['groupe'] == i):
                image = os.path.basename(base[j]['nom'])
                print(image)

#base image moments de hu pour le kmeans
def base_images_hu_moments_kmeans(base_images):
    tableau_hu=[]
    matrice_base_images_couleurs=chargement_base_image(base_images, label = True)
    taille=len(matrice_base_images_couleurs)
    
    for i in range(taille):
        temp=hu_moments(matrice_base_images_couleurs[i]['matrice'])
        
        tableau_hu.append({'nom':matrice_base_images_couleurs[i]['nom'],'hu':temp,'groupe':0})
        
    return tableau_hu

#base image histogrammes pour le kmeans
def base_images_histogramme_kmeans(base_images, echelle_reduction):
    tableau_hu=[]
    
    matrice_base_images_couleurs=chargement_base_image(base_images, label = False)
    taille=len(matrice_base_images_couleurs)
    
    for i in range(taille):
        temp=buildHistogram(matrice_base_images_couleurs[i]['matrice'],echelle_reduction)
        
        tableau_hu.append({'nom':matrice_base_images_couleurs[i]['nom'],'histo':temp,'groupe':0})
        
    return tableau_hu

#lister les fichier d'un repertoire ayant l'extension png
def liste_fichiers_repertoire(repertoire, extension_recherche):
    liste_fichiers=[]
    
    for file in os.listdir(repertoire):
        if file.endswith(extension_recherche):
            liste_fichiers.append(os.path.join(repertoire, file))
            
    return liste_fichiers

# methode permettant de regrouper toutes les options de descripteurs
def general_kmeans(base_images, k, option):
    centroids=[]
    try:
        if option == 0:
            base_histogramme=[]           
            echelle_reduction=int(input("Veuillez specifier l'échelle de réduction à apppliquer sur l'ensemble des images: "))
            #calcul des histogrammes de la base d'images
            base_histogramme=base_images_histogramme_kmeans(base_images, echelle_reduction)
            #initialisation des centroids
            centroids=initialize_centroids(k, base_histogramme, option)
            #exécution du kmeans
            perform_kmeans(base_histogramme, centroids, k, option)
            #affichage des clusters
            print_results(k, base_histogramme) 
        elif option == 1:
            base_hu=[]
            #construction des moments de hu de la base d'images
            base_hu=base_images_hu_moments_kmeans(base_images)
            #initialisation des centroids
            centroids=initialize_centroids(k, base_hu, option)
            #exécution du kmeans
            perform_kmeans(base_hu, centroids, k, option)
            #affichage des clusters
            print_results(k, base_hu)
        else:
            print("Mauvais choix\n")
            
        return centroids
    
    except Exception as e:
        print(e)

################################################################################################
# Implementation des bonus
################################################################################################
# methode de decoupage de l'image en 4 parties puis construction des histogrammes
#def couper_image(image, echelle_reduction):
#    histogram_decouper = []
#    tiles = image_slicer.slice(image, 4, save = False)
#    for m in tiles:
#         img_part = np.array(m.image)
#         histogram_decouper.extend(buildHistogram(img_part, echelle_reduction))
#        
#    return histogram_decouper

##################################################################################
# implemtation des moments de Hu
##################################################################################
# calcul des moments centres reduits de Hu
def central_hu_moments(img, p, q):
    M_00 = 0
    M_10 = 0
    M_01 = 0
    height, width = img.shape
    for x in range(height):
        for y in range(width):
            if img[x, y]>0:
                M_00 += img[x, y] 
                M_10 += x*img[x, y]
                M_01 += y*img[x, y]
    # centroid
    xx = M_10/M_00
    yy = M_01/M_00
    # calcul de n_pq
    mu_00 = M_00
    mu_pq = 0
    for ii in range(height):
        x = ii - xx
        for jj in range(width):
            if img[ii, jj]>0:
                y = jj - yy
                mu_pq += (x**p)*(y**q)*img[ii, jj]
    
    k = 0.5*(p+q)+1
    n_pq = mu_pq/mu_00**k
    
    return n_pq

# methode renvoyant les 7 moments invariants de Hu d'une image
def hu_moments(img):
    n_20 = central_hu_moments(img, 2, 0)
    n_02 = central_hu_moments(img, 0, 2)
    n_11 = central_hu_moments(img, 1, 1)
    n_30 = central_hu_moments(img, 3, 0)
    n_12 = central_hu_moments(img, 1, 2)
    n_21 = central_hu_moments(img, 2, 1)
    n_03 = central_hu_moments(img, 0, 3)
    # I_1
    I_1 = n_20 + n_02
    # I_2
    I_2 = (n_20 - n_02)**2 + 4*n_11
    # I_3
    I_3 = (n_30 - 3*n_12)**2 + (3*n_21 - n_03)**2
    # I_4
    I_4 = (n_30 + n_12)**2 + (n_21 + n_03)**2
    # I_5
    I_5 = (n_30 - 3*n_21)*(n_30 + n_12)*((n_30 + n_12)**2 - 3*(n_21 + n_03)**2) \
          + (3*n_21 - n_03)*(n_21 + n_03)*(3*(n_30 + n_12)**2 - (n_21 + n_03)**2)
    # I_6
    I_6 = (n_20 - n_02)*((n_30 + n_12)**2 - (n_21 + n_03)**2) + 4*n_11*(n_30 + n_12)*(n_21 + n_03)
    # I_7
    I_7 = (3*n_21 - n_03)*(n_30 + n_12)*((n_30 + n_12)**2 - 3*(n_21 + n_03)**2)-(n_30 + 3*n_12) \
          *(n_21 + n_03)*(3*(n_30 + n_12)**2 - (n_21 + n_03)**2)
    result = [I_1, I_2, I_3, I_4, I_5, I_6, I_7]
    
    return result

# calcul de distance des moments de Hu entre 2 images
def distance_moments_hu_img(img1, img2):
    dist = 0
    moment1 = hu_moments(img1)
    moment2 = hu_moments(img2)
    for i in range(len(moment1)):
        dist += (moment1[i] - moment2[i])**2
        
    newDistance = np.sqrt(dist)/7
                      
    return newDistance

# calcul de distance des moments de Hu entre 2 images mais avec parametres les moments
def distance_moments_hu_moments(moment1, moment2):
    dist = 0
    for i in range(len(moment1)):
        dist += (moment1[i] - moment2[i])**2
        
    newDistance = np.sqrt(dist)/7
      
    return newDistance

# methode de calcul de la similarite globale entre 2 images                         
def similarite_globale(distance_histogramme, distance_hu, poids=0.5):
    distance_globale = poids * distance_histogramme + (1-poids) * distance_hu
                                                   
    return distance_globale                                             

# methode permettant d'obtenir les moments de toute la base d'images
def base_images_hu_moments(base_images):
    tableau_hu=[]
    matrice_base_images_couleurs = chargement_base_image(base_images, label = True)
    taille=len(matrice_base_images_couleurs)
    
    for i in range(taille):
        temp=hu_moments(matrice_base_images_couleurs[i]['matrice'])
        tableau_hu.append({'nom':matrice_base_images_couleurs[i]['nom'],'hu':temp,'groupe':0})
    return tableau_hu




#image1 = Image.open('/home/admin1/Documents/INDEXATION_MULTIMODALE_des_DOCUMENTS/IFI2017/coil-100/obj1__0.png')

#req_im = input("Veuillez specifier l'image requete : ")
#k = int(input("Veuillez specifier k : "))
#liste=extraction_repertoire_extension(req_im)
#base_images=[]
#image1=[]
#base_images=lecture_base_images(liste[0], liste[1], req_im)
##chargement des images de la base
#matrice_base_images=chargement_base_image(base_images)
##calcul des histogrammes de la base d'images
#histogramme_base_images=Calcul_histogramme_base(matrice_base_images, 32)
##chargement de l'image requete
##image1 = Image.open(req_im)
##construction de l'image requete
##histogramme_image_requete=buildHistogram(image1,32)
#
##mu, cluster = kmean(histogramme_base_images, k)
#print(len(mu), len(cluster))
#calcul des k plus proches voisins
#re=knn(histogramme_base_images,k ,histogramme_image_requete)

#for i in range(k):
#    print(re[i]['nom'],re[i]['distance'])
    
#filename, extension = splitext(req_im)
#rep = os.path.dirname(req_im)
#col = load_images(rep, extension)
#img = io.imread(req_im)
#img = cv2.imread('D:\IFI\Indexation\Projet\coil-100\obj1__0.png')
#img2 = cv2.imread('D:\IFI\Indexation\Projet\coil-100\obj24__315.png')
#img_gray = to_gray(img)
#img_gray2 = to_gray(img2)
#dist = distance_moments_hu(img_gray, img_gray2)
#dist = distance_moments_hu(img, img)
#print("\nLa distance entre img et img2 est: " + str(dist))
#im = toGray(img)

#print("\nLa distance est: "+str(hu_moments(img)))


#im = misc.imread('/home/paul/Documents/Indexation/TPs/kimia216/kimia216/bird02.pgm')
#im = misc.imread('/home/paul/Documents/Indexation/TPs/coil-100/obj1__0.png')

#plt.imshow(im)