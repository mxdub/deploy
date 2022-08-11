# Projet 7 - Implémentez un modèle de scoring

Description du depôt GitHub

## Dossier API/

Code (app/main.py) et données pour l'API (https://credits-ocr-flaskapi.herokuapp.com/)

+ endpoints :
  * /get_data : renvoi l'ensemble des données non-standardisées (DataFrame au format json)
  * /get_idx : renvoi l'ensemble des ID (liste au format json)
  * /predict : renvoi l'ensemble des prédictions (DataFrame au format json)
  * /predict_one?id=XXX : renvoi la prédiction pour un client (donner ID via URL : e.g., https://credits-ocr-flaskapi.herokuapp.com/predict_one/?id=283409)
  * /get_stats : renvoi les statistiques (fn, fp, etc.) du classifieur en fonction du seuil (DataFrame au format json)
  * /get_shaps (POST request!) : renvoi les valeurs de Shapleys pour un individu (variable 'id' de la POST request HTTP)

+ données :
  * us_data_subset.csv : sous-ensemble du jeu de données
  * standard_scaler.pkl : L'objet de standardisation utilisé
  * xgb_model.pkl : Model XGBoost pré-entrainé 
  * shap_xgb_explainer.pkl : Shap explainer
 
## Dossier Dashboard/

Dashboard construit avec Dash (code app/main.py, et css associé dans le sous-dossier app/assets/) à partir d'un template pré-existant (https://dash.gallery/dash-wind-streaming/) et modifié pour le problème. Le dashboard est déployé à l'adresse suivante : https://credits-ocr-dashboard.herokuapp.com/

Il permet de sélectionner un client (Customer ID) et afficher son score (et la probabilité de défaut selon le modèle). Diverses informations sont égalements visibles : (i) l'importance locale des différentes *features* (vert : favorable, rouge : défavorable) - il est possible de choisir le nombre de *features* à présenter. (ii) les distributions de valeurs pour chacune des *features* (en fonction du status payed/default des clients).

Un onglet "Gestion du risque" est également présent, davantage à l'attention des gestionnaires clients, il permet de faire varier le seuil utilisé pour la classification ainsi que la valeur souhaité de *beta* pour le F-score (qui guide le choix du seuil). Le graphique représente les taux de faux positifs et faux négatifs, ainsi que la valeur du F-score en fonction du seuil choisi. 

## Fichier Projet7_notebook.ipynb

Notebook d'analyse (pré-traitement données - non incluses dans le dépôt par manque de place, entrainement des modèles - avec recherche sur grille + cross-validation, et interprétations globales/locales).

## Fichier Note_methodologique.md

Note méthodologique. 