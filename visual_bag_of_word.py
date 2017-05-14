# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:51:17 2017

@author: PAUL
"""
import cv2
import numpy as np
import glob
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import warnings

import utils as uts
from os.path import splitext


###############################################################################
# Implementation du dictionnaire des mots
###############################################################################

# methode renvoyant toutes les images de la base
# le chemin de chaque image
def binary_data(path, extension):
    dataset = set(glob.glob(path + '/*' + extension))
    img_paths = [path for path in dataset]

    return np.array(img_paths)

# methode renvoyant toutes les images de la base
# le chemin de chaque image
def binary_data_without_req_img(path, img_req):
    filename, extension = splitext(img_req)
    dataset = set(glob.glob(path + '/*' + extension))
    img_paths = [path for path in dataset if path != img_req]

    return np.array(img_paths)
    
# repartition des donnees en apprentissage, test et validation
# 2/3 pour apprentissage, 1/6 pour test et 1/6 pour validation
def train_test_val_split_idxs(total_rows, split):
    # pourcentage de test
    percent_test = 1.0 - split
    # pourcentage de validation du modele
    percent_val = 1.0 - split
    
    row_range = range(total_rows)
    # recuperation aleatoire des donnees de test, d'apprentissage et de validation
    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows)
    # supprimer les indexes de test
    row_range = [idx for idx in row_range if idx not in test_idxs]
    no_val_rows = int(total_rows*(percent_val))
    val_idxs = np.random.choice(row_range, size=no_val_rows)
    # supprimer les indexes de validation
    training_idxs = [idx for idx in row_range if idx not in val_idxs]
    print ("\n** Subdivision de la base: apprentissage = %i, test = %i, validation = %i **\n"\
           % (len(training_idxs), len(test_idxs), len(val_idxs)))
    
    return training_idxs, test_idxs, val_idxs

# methode de generation des points d'interets et des descripteurs    
def gen_sift_features(img_paths):
    img_keypoints = {}
    img_descs = []
    tot_desc = 0
    print ("\\** Generation des descripteurs SIFT pour %i images **\n" % len(img_paths))
    for img_path in img_paths:
        img = uts.read_image(img_path)
        # conversion de l'image en niveau de gris
        gray_img = uts.to_gray(img)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray_img, None)
        img_keypoints[img_path] = kp
        img_descs.append(desc)
        tot_desc += len(desc)

    print ("** %i Descripteurs SIFT generes **\n" % tot_desc)
    
    return img_descs

# classification des descripteurs
def cluster_features(img_descs, training_idxs, cluster_model):
    n_clusters = cluster_model.n_clusters
    # Concatenation des descripteurs d'apprentissage a partir de leur index
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs\
                             for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError("Les descripteurs SIFT devraient avoir 128 caracteristiques, au lieu de ",\
                         all_train_descriptors.shape[1])

    print ("\n%i descripteurs avant le clustering" % all_train_descriptors.shape[0])

    # Clustering des descripteurs pour obtenir le dictionnaire
    print ("\n** Modele de classification utilise: %s **\n" % repr(cluster_model))
    print ("Clustering sur l'ensemble d'apprentissage pour obtenir un dict. de %i mots\n" % n_clusters)

    # apprentissage avec k-means sur ces descripteurs ci-dessus selectionnes
    cluster_model.fit(all_train_descriptors)
    print ("** Clustering effectue. Utilisation du modele de clustering \npour generer les histogrammes BoW pour chaque image. **")

    # calcul de l'ensemble de mots reduits de chaque image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # construction de l'histogramme des caracteristiques de chaque image
    img_bow_hist = np.array([np.bincount(clustered_words, minlength=n_clusters)\
                             for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ("\n** Generation des histogrammes BoW effectuee. **")
    
    return X, cluster_model

# amelioration de la separation des donnees
def perform_data_split(X, training_idxs, test_idxs, val_idxs):
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    X_val = X[val_idxs]

    return X_train, X_test, X_val

# classification d'une image a partir du modele obtenu de k-means
# en parametre: l'image et le modele de classification
def img_to_hist(img_path, cluster_model, X_train):
    print ("\n** Classification de l'image %s **" % (os.path.basename(img_path)))
    img = uts.read_image(img_path)
    gray = uts.to_gray(img)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = cluster_model.predict(desc)
    
    accuracy = metrics.accuracy_score(clustered_desc, X_train)
    print ("Precision de classification: %s" % "{0:.3%}".format(accuracy))
    
    #score_ = cluster_model.score(desc)
    #print("Precision de classification: %2f" % (score_))
    
    # frequence d'apparition des mots dans le dictionnaire
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # l'histogramme de l'image classifiee a partir du modele des mots visuels
    return img_bow_hist.reshape(1,-1)
    
# classification avec k-means
def cluster_and_split(img_descs, training_idxs, test_idxs, val_idxs, k):
    warnings.filterwarnings('ignore')

    X, cluster_model = cluster_features(
        img_descs,
        training_idxs=training_idxs,
        cluster_model=MiniBatchKMeans(n_clusters = k)
    )

    warnings.filterwarnings('default')

    X_train, X_test, X_val = perform_data_split(X, training_idxs, test_idxs, val_idxs)
    
    accuracy = metrics.accuracy_score(val_idxs, training_idxs)
    print ("Precision du test : %f"%(accuracy))

    return X_train, X_test, X_val, cluster_model    