#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:44:26 2024

@author: vladimirfernandez
"""

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt




def load_housing_data():
    """
        Metodo para descargar dataset housing
    """
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """
    Guardar imagenes en directorio images
    """
    
    IMAGES_PATH = Path() / "images" / "end_to_end_project"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
      
    