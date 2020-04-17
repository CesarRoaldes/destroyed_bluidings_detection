#!/home/cesar/anaconda3/envs/sat/bin/python
import argparse
from data_download import *
from pred import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("min_lat", help="Lattitude minimum de la zone ciblée (int)")
    parser.add_argument("min_lon", help="Longitude minimum de la zone ciblée (int)")
    parser.add_argument("max_lat", help="Lattitude maximum de la zone ciblée (int)")
    parser.add_argument("max_lon", help="Longitude maximum de la zone ciblée (int)")
    parser.add_argument("description", help="Description de la zone ciblée (string sans espace)")
    
    args = parser.parse_args()
    
    min_lon, min_lat = float(args.min_lon), float(args.min_lat)
    max_lon, max_lat = float(args.max_lon), float(args.max_lat)
    description = args.description
    
    # Download
    setup()
    # download_images(min_lon, min_lat, max_lon, max_lat, description)

    # Predictions
    # run_model(description)

    # Plot
    make_map(description)