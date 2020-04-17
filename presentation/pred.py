import os
import sys
import time
import copy
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
from PIL import Image
from shapely import wkt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

DATA_DIR = './data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
PROCESS_DIR = os.path.join(DATA_DIR, 'process')
MODEL_DIR  = os.path.join(DATA_DIR, 'models')
FIGURE_DIR  = os.path.join(DATA_DIR, 'figure')


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
def get_input_transform():   
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    return transf(img).unsqueeze(0)

def prediction_image_path(image_path, model, device):
    """ 
        Sort les predictions d'un model sur l'image
        passee en arguement.
    """
        
    img = get_image(image_path)
    
    model = model.to(device)
    model.eval()
    
    img_t = get_input_tensors(img)
    img_t = img_t.to(device)
    
    output = model(img_t)
    _, pred = torch.max(output, 1)
    
    pred = pred.detach().cpu().numpy()[0]
    
    return pred

def run_model(description):
    
    print("Déploiement du modèle")
    
    #set up le GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('%d GPU(s) disponible.' % torch.cuda.device_count())
        print('GPU utilisé :', torch.cuda.get_device_name(0))

    else:
        print('Pas de GPU disponible, utilisation du CPU.')
        device = torch.device("cpu")
    
    #charge le dataset
    
    data_preds = pd.read_csv(os.path.join(
        PROCESS_DIR, 'images_label_'+ description + '.csv'))
    
    #charge le modele
    
    model_euroSAT = models.resnet50(pretrained=False, progress=False)
    num_ftrs = model_euroSAT.fc.in_features
    model_euroSAT.fc = nn.Linear(num_ftrs, 2)
    model_euroSAT.load_state_dict(torch.load(os.path.join(MODEL_DIR,
                                                          "all_model_euroSAT.pt")))
    
    #run le modele
    pred_euroSAT = []
    print("\nDébut de la phase de prédiction")
    for idx, obs in data_preds.iterrows():
        image_nom = obs['nom_images_torchvision']
        image_path = os.path.join(IMAGES_DIR, description, image_nom)
        pred_euroSAT.append(prediction_image_path(image_path, model_euroSAT, device))
        sys.stdout.write("\r[" + "#" * (int(((idx + 1) / data_preds.shape[0]) * 100)) \
                         + " " * (int(((data_preds.shape[0] - (idx + 1)) / data_preds.shape[0]) * 100)) \
                         + "] %d%%" % int(idx / data_preds.shape[0]*100))
        sys.stdout.flush()
        time.sleep(0.01)
    sys.stdout.write("\r[" + "#" * 100 + "] 100%")                
    sys.stdout.write("\n")
        
    data_preds['pred_euroSAT'] = pred_euroSAT
    
    print("Toutes les images ont été traitées... Opération terminée.\n")
    
    #sauvegarde le modèle
    data_preds.to_csv(os.path.join(PROCESS_DIR, 'images_label_' + \
                                                description + \
                                                '.csv'),
                      index=False)
    
    return 0

def make_map(description):

    data_preds = pd.read_csv(os.path.join(
        PROCESS_DIR, 'images_label_'+ description + '.csv'))
    
    data_preds['geometry'] = data_preds['centroide']
    data_preds['Coordinates'] = data_preds['centroide'].apply(wkt.loads)
    # data_preds = data_preds.set_geometry('Coordinates')
    data_preds = gpd.GeoDataFrame(data_preds, geometry='Coordinates')
    
    print("Création de la carte")
    
    # plt.figure(figsize=(10, 10))
    
    ax = gplt.webmap(data_preds, projection=gcrs.WebMercator(), figsize=(14,14))
    gplt.kdeplot(data_preds[data_preds['pred_euroSAT'] == 1], cmap='RdYlGn_r', n_levels=30,
                 shade=True, shade_lowest=False, ax=ax, alpha=0.2,
                 label='Predicted density of destroyed buildings')

    plt.legend(fontsize=5, loc = "lower right")
    plt.title('Models predictions on ' + description, fontsize=10)
    print('allo')
    plt.savefig(os.path.join(FIGURE_DIR, 'model_prediction_' + description + '.jpg'),
                optimize=True, quality=95)
    print("Le mapping des destructions sur " + description + " est enregistré dans le dossier '{}' !\n\n".format(
        IMAGES_DIR_DESCRIPTION))
    
    return 0