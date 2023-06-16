# Importer les bibliothèques nécessaires
import streamlit as st 
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import timedelta, date
start_date = date(2020, 5, 1)
end_date = date(2020, 5, 3)
delta = timedelta(days=1)

url = "https://covid-19-statistics.p.rapidapi.com/reports"
headers = {
    'x-rapidapi-key': "1d0664840fmsh6d34b986315527bp1d5700jsn7cc6123956a8",
    'x-rapidapi-host': "covid-19-statistics.p.rapidapi.com"
}

data_frames = []
current_date = start_date

# Générer une séquence de dates entre start_date et end_date

dates = pd.date_range(start_date, end_date, freq='D')

# Itérer sur la séquence de dates
for current_date in dates:
    print(f'Retrieving data for {current_date}')
    querystring = {"date": current_date.strftime('%Y-%m-%d')}
    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    df = pd.DataFrame.from_dict(data['data'])
    data_frames.append(df)

result = pd.concat(data_frames, ignore_index=True)
print(result.head())
result.info()
result = result.dropna()
print(result)
result['region'] = result['region'].astype(str)
result = result.drop_duplicates()
result.info()


sns.displot(result['confirmed'])
plt.show()

sns.scatterplot(data=result, x='confirmed', y='deaths')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='recovered')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='active')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='fatality_rate')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='confirmed_diff')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='deaths_diff')
plt.show()

sns.scatterplot(data=result, x='confirmed', y='recovered_diff')
plt.show()

columns_to_drop = ['last_update', 'fatality_rate']
result = result.drop(columns_to_drop, axis=1)

numeric_columns = result.select_dtypes(include='number').columns
corr = result[numeric_columns].corr()
sns.heatmap(corr, annot=True)
plt.show()


# Régression linéaire

from sklearn.preprocessing import StandardScaler

# Sélection des colonnes à utiliser comme variables indépendantes
X = result[['deaths', 'recovered', 'confirmed_diff', 'deaths_diff', 'active']]
# Sélectionn de la colonne à utiliser comme variable dépendante
y = result['confirmed']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Création d'un objet StandardScaler
scaler = StandardScaler()

# Normalisation des données d'entraînement
X_train_scaled = scaler.fit_transform(X_train)

# Entraînement d'un modèle de régression linéaire sur les données normalisées
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Normalisation des données de test
X_test_scaled = scaler.transform(X_test)

# Évaluation des performances du modèle sur l'ensemble de test
score = model.score(X_test_scaled, y_test)
print(f"R²: {score}")

from sklearn.metrics import mean_squared_error
# Faites des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Calculez l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# validation croisée et recherche de grille pour optimiser les performances du modèle.

from sklearn.model_selection import GridSearchCV, cross_val_score

# Définition des hyperparamètres à tester
param_grid = {'fit_intercept': [True, False]}

# Création d'un objet GridSearchCV
grid = GridSearchCV(LinearRegression(), param_grid, cv=5)

# Entraînement du modèle en utilisant la recherche de grille
grid.fit(X_train_scaled, y_train)

# Affichage des meilleurs hyperparamètres
print(f"Best hyperparameters: {grid.best_params_}")

# Évaluation des performances du modèle en utilisant la validation croisée
scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
print(f"Cross-validation scores: {scores}")

# Création d'une interface utilisateur avec Streamlit
st.title('Prédiction du nombre de cas COVID-19')

# Création des champs de saisie pour les variables indépendantes
deaths = st.number_input('Nombre de décès', value=0)
recovered = st.number_input('Nombre de guérisons', value=0)
confirmed_diff = st.number_input('Différence de cas confirmés', value=0)
deaths_diff = st.number_input('Différence de décès', value=0)
active = st.number_input('Nombre de cas actifs', value=0)

# Création d'un bouton pour effectuer la prédiction
if st.button('Prédire'):
    # Création d'un DataFrame avec les données saisies par l'utilisateur
    input_data = pd.DataFrame({
        'deaths': [deaths],
        'recovered': [recovered],
        'confirmed_diff': [confirmed_diff],
        'deaths_diff': [deaths_diff],
        'active': [active]
    })

    # Normalisation des données saisies par l'utilisateur
    input_data_scaled = scaler.transform(input_data)

    # Faire une prédiction avec le modèle entraîné
    prediction = model.predict(input_data_scaled)

    # Affichage la prédiction
    st.write(f'Prédiction du nombre de cas : {prediction[0]:.0f}')



















