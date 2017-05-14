#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:01:21 2017

@author: paul
"""
import os
import visual_bag_of_word as vbow
import image_comparison as im_com
import utils as uts
import glob

from sklearn.externals import joblib
from os.path import splitext

def display_welcome():
    print("*******************************************************************")
    print("*        Bienvenue dans l'utilitaire d'indexation d'images        *")
    print("*******************************************************************")
    
def display_menu():
    print("** Descripteurs globaux **")
    print("\t1-Histogramme couleur")
    print("\t2-Les moments de Hu")
    print("\t3-KNN")
    print("\t4-Clustering K-Means")
    print("** Descripteurs lobaux et Bag-of-Words **")
    print("\t5-Clustering K-Means")
    print("\t6-Classification d'une image")
    print("** Bonus **")
    print("\t7-Plusieurs histogrammes couleurs par image")
    print("*******************************************************************")
    
# methode principale d'execution de tout le programme
def main():
    display_welcome()
    rep = "o"
    while rep.upper() != "N":
        display_menu()
        option = int(input("Veuillez choisir une option: "))
        if option == 1:
            req_im = input("Veuillez specifier l'image requete: ")
            echelle = int(input("Veuillez specifier l'echelle de reduction: "))
            img = uts.read_image(req_im)
            print("Calcul de l'histogramme")
            histo = im_com.buildHistogram(img, echelle)
            print("Affichage de l'histogramme")
            uts.show_histo(histo)
        elif option == 2:
            req_im = input("Veuillez specifier l'image requete: ")
        elif option == 3:
            print("\t0-Avec histogramme")
            print("\t1-Avec les moments de Hu")
            print("\t2-Avec la distance globale de similarite")
            
            choice = int(input("Veuillez specifier votre choix: "))
            req_im = input("Veuillez specifier l'image requete: ")
            
            liste = im_com.extraction_repertoire_extension(req_im)
            base_images = im_com.lecture_base_images(liste[0], liste[1], req_im)
            # appel de la methode de KNN
            im_com.knn(base_images, req_im, choice)
                
        elif option == 4:
            print("\t0-distance histogramme")
            print("\t1-distance de moments de Hu")
            
            choix = int(input("Veuillez specifier votre choix: "))
            rep = input("Veuillez specifier le repertoire de la base d'image: ")
            extension = input("Veuillez specifier l'extension des images: ")
            k = int(input("Veuillez specifier le nombre de clusters k: "))
            
            base = im_com.liste_fichiers_repertoire(rep, extension)
            # appel de la methode k-means
            im_com.general_kmeans(base, k, choix)
            # clustering avec bag-of-words
        elif option == 5:
            base = input("Veuillez specifier le repertoire de la base d'image: ")
            extension = input("Veuillez specifier l'extension des images: ")
            k = int(input("Veuillez specifier le nombre de clusters k: "))
            
            #path = os.path.dirname(base)
            img_paths = vbow.binary_data(base, extension)
            img_descs = vbow.gen_sift_features(img_paths)
            # generation d'indexes pr apprentissage/test et validation
            training_idxs, test_idxs, val_idxs = vbow.train_test_val_split_idxs(total_rows=len(img_descs), split=0.67)
            results = {}
            X_train, X_test, X_val, cluster_model = vbow.cluster_and_split(img_descs, training_idxs, test_idxs, val_idxs, k)
            
            print ("\nCentre de gravite du clustering avec k = %i est: " % k, cluster_model.inertia_)
            
            results[k] = dict(inertia = cluster_model.inertia_, cluster_model=cluster_model)
            
            print ("\n*** k = %i Fini ***\n" % k)
            # sauvegarde dans un fichier pr des analyses ulterieures
            ################################################################################################
            feature_data_path = 'pickles/k_grid_feature_data/'
            result_path = 'pickles/k_grid_result'
            
            # supprimer l'ancienne sauvegarde
            for path in [feature_data_path, result_path]:
                for f in glob.glob(path+'/*'):
                    os.remove(f)
            
            print ("** Sauvegarde des donnees separees **")
            for obj, obj_name in zip( [X_train, X_test, X_val],
                                     ['X_train', 'X_test', 'X_val'] ):
                joblib.dump(obj, '%s%s.pickle' % (feature_data_path, obj_name))
            
            print ("** Sauvegarde du resultats **")
            exports = joblib.dump(results, '%s/result.pickle' % result_path)
            
            print ('\n\t\t\t*** Resultat ***')
            print ("Pour k = %i:\tCentre de gravite du K-Means est %f" % (k, results[k]['inertia']))
            
        elif option == 6:
            req_im = input("Veuillez specifier l'image requete: ")
            k = int(input("Veuillez specifier le nombre de clusters k: "))
            filename, extension = splitext(req_im)
            path = os.path.dirname(req_im)
            img_paths = vbow.binary_data_without_req_img(path, req_im)
            img_descs = vbow.gen_sift_features(img_paths)
            # generation d'indexes pr apprentissage/test et validation
            training_idxs, test_idxs, val_idxs = vbow.train_test_val_split_idxs(total_rows=len(img_descs), split=0.67)
            
            X_train, X_test, X_val, kmeans = vbow.cluster_and_split(img_descs, training_idxs, test_idxs, val_idxs, k)
            freq = vbow.img_to_hist(req_im, kmeans)
            #print(os.path.basename(req_im), freq)
            uts.show_histo(freq)
            
        elif option == 7:
            req_im = input("Veuillez specifier l'image requete: ")
        else:
            print("Mauvaise option")
            
        rep = input("\nVoulez-vous continuer d'utiliser cet utilitaire? (o/n): ")
        print("\n")
    else:
        pass
    
 # D:\IFI\Indexation\Projet\base\obj1__0.png  
#    req_im = input("Veuillez specifier l'image requete : ")
#    filename, extension = splitext(req_im)
#    path = os.path.dirname(req_im)
#    img_paths = vbow.binary_data(path, extension)
#    img_descs = vbow.gen_sift_features(img_paths)
#    
#    # generation d'indexes pr apprentissage/test et validation
#    training_idxs, test_idxs, val_idxs = vbow.train_test_val_split_idxs(total_rows=len(img_descs), split=0.67)
#    
#    results = {}
#    k_vals = [50, 100, 150, 300]
#    
#    for k in k_vals:
#        X_train, X_test, X_val, cluster_model = vbow.cluster_and_split(
#            img_descs, training_idxs, test_idxs, val_idxs, k)
#    
#        print ("\nCentre de gravite du clustering avec k = %i est: " % k, cluster_model.inertia_)
#    
#        results[k] = dict(inertia = cluster_model.inertia_, cluster_model=cluster_model)
#    
#        print ("\n*** k = %i Fini ***\n" % k)
#    
#    print ("*******************************")
#    print ("***** Tous les k termines *****")
#    print ("*******************************\n")
#    
#    # sauvegarde dans un fichier pr des analyses ulterieures
#    ################################################################################################
#    
#    feature_data_path = 'pickles/k_grid_feature_data/'
#    result_path = 'pickles/k_grid_result'
#    
#    # supprimer l'ancienne sauvegarde
#    for path in [feature_data_path, result_path]:
#        for f in glob.glob(path+'/*'):
#            os.remove(f)
#    
#    print ("** Sauvegarde des donnees separees **")
#    
#    for obj, obj_name in zip( [X_train, X_test, X_val],
#                             ['X_train', 'X_test', 'X_val'] ):
#        joblib.dump(obj, '%s%s.pickle' % (feature_data_path, obj_name))
#    
#    print ("** Sauvegarde du resultats **")
#    
#    exports = joblib.dump(results, '%s/result.pickle' % result_path)
#    
#    print ('\n\t\t\t*** Resultat ***')
#    
#    k_vals = sorted(results.keys())
#    for k in k_vals:
#        print ("Pour k = %i:\tCentre de gravite du K-Means est %f" % (k, results[k]['inertia']))
      

# /home/paul/Documents/Indexation/Projet/base/obj1__0.png    
    
    
if __name__ == '__main__':
    main()
