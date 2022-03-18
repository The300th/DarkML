# DarkML
Use of Machine Learning for finding a mapping between dark-matter-only simulations and hydrodinamic simulations. We have populated 3 dark-matter-only simulations with baryon properties: the Multidark Planck simualtion (MDPL2)  box_size=1 Gpc h^{-1} and the UNIT simulation box_size = 1Gpc h^{-1}  with 2048^{3} (UNITSIM2048) and 4096^{3} (UNITSIM4096) particles. We have uploaded csv files containing the generated data and also the trained XGBoost model for the user's convenience. 
## dependences
* [XGBoost](https://xgboost.readthedocs.io/en/stable/) 
* [scikit-learn](https://xgboost.readthedocs.io/en/stable/)
* [pandas](https://pandas.pydata.org/)

## Baryon Catalogs
We have generated baryon catalogs for the three DM-only simulations listed above. Below, we provide a link to download the link of the catalogs.
* [MDPL2](https://dauam-my.sharepoint.com/:x:/g/personal/daniel_deandres_uam_es/EcyULkS8XpRKm_INV5dlS1EB_rUOpqOCbeDAg1DXG5-jiA?e=1UiD8y)
* [UNITSIM2048](https://dauam-my.sharepoint.com/:x:/g/personal/daniel_deandres_uam_es/EbjfLOfjKSBEotpHDyWLQW4BLIpoISDEqMbdhCZZejkJwg?e=wBNlci).
* [UNITSIM4096](https://dauam-my.sharepoint.com/:x:/g/personal/daniel_deandres_uam_es/EVl9JeU00bRJixIyzChHDjQB234Pw9LEhjg7gGHBu3sTDA?e=6MqCGs)

To read the data products you can use pandas: `pandas.read_csv(file_path)`. We provide the 5 baryonic predicted properties. Note that the logarithmic value is used for the properties. For more information, we refer to the paper [link](in progress)


## Generate your own catalog
We have created a function that reads a file containing a list of the most relevant features and predicts the baryon properties. The function reads the save models in the desired `model_path` and the data frame containing all the DM-only properties listed in the file `feature_file`  and predicts the properties for all the targets inside `feature_file`. The predictions is the average of the 10 models considered in the 10K-fold cross-validation.

```
import numpy as np
import sys
import os
import pickle
import json

def xgb_pred(target,df):
    
    with open(feature_file, 'r') as fp:
        final_features = json.load(fp)
    preds = []
    
    for kfold in range(10):
        model_file = model_path + 'redXGBkfolds_{}/models/model_kfold{}.sav'.format(target,kfold)
        model = pickle.load(open(model_file, 'rb'))
        features = final_features[target]
        preds.append(model.predict(df[features].values))

    pred = np.array(preds).mean(axis=0)
    
    return pred
```
