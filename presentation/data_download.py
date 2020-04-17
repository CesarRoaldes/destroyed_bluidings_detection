import os
import sys
import time
import math
import requests
import urllib
from urllib import request 
from PIL import Image
import io
from io import BytesIO
from math import cos

import scipy
import imageio
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, osr
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point, Polygon

BASE_DIR = './'
IMAGES_DIR = os.path.join(BASE_DIR, "data/images")
DATA_DIR = os.path.join(BASE_DIR, "data/process")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")
FIGURE_DIR  = os.path.join(BASE_DIR, "data/figure")

def setup():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return 0

def save_google_map(file_name, url):
        '''
            La fonction enregistre les images. 
        '''
        buffer = BytesIO(request.urlopen(url).read())
        image = imageio.imread(buffer, pilmode='RGB')
        imageio.imwrite(file_name, image)

def creation_zone(point, m_px):
        """ 
            Defini les coordonnees du point en bas a droite de l'image
            et les coordonnees du point en haut a gauche pour ensuite creer
            le polygone cree par l'image associe au point (zoom 19 seulement).
            
            Retourne un tuple de (lon, lat) dans l'ordre (bg, bd, hd, hg).
        """
        lat, lon = point[0], point[1]
                                                        
        lon_bg = lon - (m_px * 300) / ((40075 * 1000) * cos(0) / 360)
        lon_hd = lon + (m_px * 300) / ((40075 * 1000) * cos(0) / 360)
                                                            
        lat_bg = lat - (m_px * 300) / (111.32 * 1000)
        lat_hd = lat + (m_px * 300) / (111.32 * 1000)
                                                                                    
        return ((lat_bg, lon_bg), (lat_bg, lon_hd), (lat_hd, lon_hd), (lat_hd, lon_bg))

def image_associee(description, coord_img):
        """ 
            Associe a chaque coordonee d'image le nom de
            l'image telechargee correspondante.
        """
        lat = coord_img[0]
        lon = coord_img[1]
        file_name = description + '_' + str(lon) + '_' + str(lat) + '.jpg'
        return file_name

def download_images(min_lon, min_lat, max_lon, max_lat, description):
    
    zoom = 19
    size = 600
    m_px = 0.274877906944
    
    cle_api_file = open(os.path.join(BASE_DIR, "cle_api.txt"), "r")
    key = cle_api_file.readline()
    print("Clé API utilisée : {}".format(key))

    liste_lon = []
    liste_lat = []

    lat = min_lat
    lon = min_lon

    while lat < max_lat:
        liste_lat.append(lat)
        lat += (m_px * 600) / (111.32 * 1000)

    while lon < max_lon:
        liste_lon.append(lon)
        lon += (m_px * 600) / ((40075 * 1000) * abs(cos(0)) / 360)

    ############################################################################
    coord_images = []
    for x in liste_lat:
        for y in liste_lon:
            if x < min_lat or x > max_lat:
                pass
            elif y < min_lon or y > max_lon:
                pass
            else:
                coord_images.append((x, y))
    print("Nombre d'images candidates : {} \n".format(len(coord_images)))
    #############################################################################
    
    shaply_coord = [Point(y, x) for x, y in coord_images]
    zone_images = [creation_zone(point, m_px) for point in coord_images]

    print("Chaque image a été traitée : {}\nNous possédons {} zones.".format(
        len(zone_images) == len(coord_images), len(zone_images)))
    
    ################################################################################
    
    #Telechargement des images
    size_str = '600x600'
    IMAGES_DIR_DESCRIPTION = os.path.join(IMAGES_DIR, description)   
    os.makedirs(IMAGES_DIR_DESCRIPTION, exist_ok=True)
  
    ##################################################################################
    m = 0 # compteur d'images

    for coord in coord_images:
        lat = coord[0]
        lon = coord[1]
        url = 'https://maps.googleapis.com/maps/api/staticmap?center=' \
            + str(lat) + ',' + str(lon) + '&zoom='+ str(19) \
            + '&size='+ size_str + '&maptype=satellite&key=' + key 
        file_name = description+ '_' + str(lon) + '_' + str(lat) +'.jpg'
        save_google_map(os.path.join(IMAGES_DIR_DESCRIPTION, file_name), url)
        sys.stdout.write("\r[" + "#" * (int(((m + 1) / len(zone_images)) * 100)) \
                         + " " * (int(((len(zone_images) - (m + 1)) / len(zone_images)) * 100)) \
                         + "] %d%%" % int(m / len(zone_images)*100))
        sys.stdout.flush()
        time.sleep(0.005)
        m += 1
        
    sys.stdout.write("\r[" + "#" * 100 + "] 100%")                
    sys.stdout.write("\n")
        
    
    print("Nombre d'images téléchargées : {}".format(m))
    
    ###################################################################################   
    df_images = pd.DataFrame()
    df_images["centroide"] = coord_images
    df_images["zone"] = [tuple([(point[1], point[0]) for point in poly]) for poly in zone_images]
    df_images["image"] = df_images["centroide"].apply(
        lambda x: image_associee(description, x))
    df_images["centroide"] = df_images["centroide"].apply(lambda x: Point(x[1], x[0]))
    df_images["zone"] = df_images["zone"].apply(lambda x: Polygon(x))
    df_images["label"] = [0 for _ in range(df_images.shape[0])]
    
    nom_images_torchvision = []
    for i in range(df_images.shape[0]):
        nom_images_torchvision.append(description + '_' + str(i).zfill(4) + '.jpg')
    df_images['nom_images_torchvision'] = nom_images_torchvision
    
    for i in range(df_images.shape[0]):
        ancien_nom_image = df_images.loc[i, "image"]
        nouveau_nom_image = df_images.loc[i, "nom_images_torchvision"]
        os.rename(os.path.join(IMAGES_DIR_DESCRIPTION, ancien_nom_image),
                  os.path.join(IMAGES_DIR_DESCRIPTION, nouveau_nom_image))
    
    df_images.to_csv(os.path.join(DATA_DIR, "images_label_" + description + ".csv"),
                     index=False)
    
    print("Les images sont enregistrées dans le dossier '{}' !\n\n".format(
        IMAGES_DIR_DESCRIPTION))
    
    return 0