"""
Created on Fri May 17 22:00:32 2024

@author: vladimirfernandez
"""
# %% 
#* 1. Importar librerias
from utils import load_housing_data
from plots import *
from  sklearn.model_selection import train_test_split
from utils import save_fig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from combined_attributes_adder import CombinedAttributesAdder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# %% 
#* 2. Load Dataset
housing = pd.read_csv("../data/housing.csv")

# %% 
#* 3. EXPLORAR LOS DATOS

# Info columnas dataset
housing.info()

# Estadisticas datsets
stast = housing.describe()

# Muestreo Estratificado
bins=[0., 1.5, 3.0, 4.5, 6., np.inf]
labels = [1, 2, 3, 4, 5]
housing["income_cat"] = pd.cut(housing["median_income"], bins=bins, labels=labels)


# %% 
#* 4. SEPARAR DATOS DE ENTRENAMIENTO Y PRUEBA
strat_train_set, strat_test_set = train_test_split(housing, 
                                                   test_size=0.2, 
                                                   stratify=housing["income_cat"], 
                                                   random_state=42)

# eliminar atributo income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# %% 
#* 5. PREPARAR LOS DATOS PARA LOS ALGORITMOS
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# Elimine el atributo de texto porque la mediana solo se puede calcular en atributos numéricos:
housing_num = housing.drop("ocean_proximity", axis=1)

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room=True)),
        ('std_scaler', StandardScaler())
    ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# %%
#? SELECCIONAR MODELOS PROMETEDORES
#* 6. ENTRENAR MODELO LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# %% 
#* 6.1 EVALUAR MODELO LinearRegression

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions", lin_reg.predict(some_data_prepared))
print("Labels", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_rmse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_rmse)
lin_rmse

# %% 
#* 7. ENTRENAR MODELO DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
# %% 
#* 7.1 EVALUAR MODELO DecisionTreeRegressor

housing_predictions = tree_reg.predict(housing_prepared)
tree_rmse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_rmse)
tree_rmse


# %%
#* 8 EVALUACION MEJOR UTILZANDO VALIDACION CRUZADA DecisionTreeRegressor
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# %%
#* 8.1 EVALUACION MEJOR UTILZANDO VALIDACION CRUZADA LinearRegression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
# %%
#* 9. ENTRENAR MODELO RandomForestRegresor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# %%
#* 9.1 EVALUAR MODELO RandomForestRegresor
housing_predictions = forest_reg.predict(housing_prepared)
forest_rmse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_rmse)
forest_rmse
# %%
#* 9.2 EVALUACION MEJOR UTILIZANDO VALIDACION CRUZADA RandomForestRegresor
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
# %%
#* PERFECIONANDO EL MODELO BUSQUEDA EXHAUSTIVA

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)