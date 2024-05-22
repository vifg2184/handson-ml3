#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:49:13 2024

@author: vladimirfernandez
"""
from utils import save_fig
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
from pandas.plotting import scatter_matrix



def create_histogram(dataset, name="attribute_histogram_plots"):
    """Histograma de cada columna numerica"""
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    dataset.hist(bins=50, figsize=(12, 8))
    save_fig(name)
        
    
def income_category(dataset, name):
    """Visualizar categoria de ingresos"""
    dataset["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    save_fig(name)
        
             
def visualizing_geo1(dataset, name="bad_visualization_plot"):
   """Visualizar datos georgraficos"""
   dataset.plot(kind="scatter", x="longitude", y="latitude", grid=True)
   save_fig(name)
        
    
def visualizing_geo2(dataset, name="better_visualization_plot"):
    """Visualizar datos georgraficos"""
    dataset.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    save_fig(name)
    
    
def visualizing_geo3(dataset, name="housing_prices_scatterplot"):
     """"""
     dataset.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=dataset["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
     save_fig(name)
        
        
def visualizing_geo_with_image(dataset, name="california_housing_prices_plot"):
    """"""
    IMAGES_PATH = Path() / "images" / "end_to_end_project"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        
    # Download the California image
    filename = "california.png"
    if not (IMAGES_PATH / filename).is_file():
        homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
        url = homl3_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, IMAGES_PATH / filename)

    housing_renamed = dataset.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
        
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))

    california_img = plt.imread(IMAGES_PATH / filename)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)
    save_fig(name)
    
    
def visualizing_correlation(dataset):
    """"""
    attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
    scatter_matrix(dataset[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")  # extra code    
    

    
    
        
    