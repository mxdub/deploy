# Projet 7 - Impl�mentez un mod�le de scoring

Description du dep�t GitHub

## Dossier API/

Code (app/main.py) et donn�es pour l'API (https://credits-ocr-flaskapi.herokuapp.com/)

+ endpoints :
  * /get_data : renvoi l'ensemble des donn�es non-standardis�es (DataFrame au format json)
  * /get_idx : renvoi l'ensemble des ID (liste au format json)
  * /predict : renvoi l'ensemble des pr�dictions (DataFrame au format json)
  * /predict_one?id=XXX : renvoi la pr�diction pour un client (donner ID via URL : e.g., https://credits-ocr-flaskapi.herokuapp.com/predict_one/?id=283409)
  * /get_stats : renvoi les statistiques (fn, fp, etc.) du classifieur en fonction du seuil (DataFrame au format json)
  * /get_shaps (POST request!) : renvoi les valeurs de Shapleys pour un individu (variable 'id' de la POST request HTTP)

+ donn�es :
  * us_data_subset.csv : sous-ensemble du jeu de donn�es
  * standard_scaler.pkl : L'objet de standardisation utilis�
  * xgb_model.pkl : Model XGBoost pr�-entrain� 
  * shap_xgb_explainer.pkl : Shap explainer
 
## Dossier Dashboard/

Dashboard construit avec Dash (code app/main.py, et css associ� dans le sous-dossier app/assets/) � partir d'un template pr�-existant (https://dash.gallery/dash-wind-streaming/) et modifi� pour le probl�me. Le dashboard est d�ploy� � l'adresse suivante : https://credits-ocr-dashboard.herokuapp.com/

Il permet de s�lectionner un client (Customer ID) et afficher son score (et la probabilit� de d�faut selon le mod�le). Diverses informations sont �galements visibles : (i) l'importance locale des diff�rentes *features* (vert : favorable, rouge : d�favorable) - il est possible de choisir le nombre de *features* � pr�senter. (ii) les distributions de valeurs pour chacune des *features* (en fonction du status payed/default des clients).

Un onglet "Gestion du risque" est �galement pr�sent, davantage � l'attention des gestionnaires clients, il permet de faire varier le seuil utilis� pour la classification ainsi que la valeur souhait� de *beta* pour le F-score (qui guide le choix du seuil). Le graphique repr�sente les taux de faux positifs et faux n�gatifs, ainsi que la valeur du F-score en fonction du seuil choisi. 

## Fichier Projet7_notebook.ipynb

Notebook d'analyse (pr�-traitement donn�es - non incluses dans le d�p�t par manque de place, entrainement des mod�les - avec recherche sur grille + cross-validation, et interpr�tations globales/locales).

## Fichier Note_methodologique.md

Note m�thodologique. 