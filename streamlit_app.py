import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Suppression des avertissements
warnings.filterwarnings("ignore")

# ==============================================
# Configuration de la Page
# ==============================================
st.set_page_config(layout="wide")

# ==============================================
# Internationalisation (i18n)
# ==============================================
languages = {
    'Français': {
        'title': '📈 Tableau de Bord des Marchés Financiers Amélioré',
        'language_selection': 'Sélectionnez la langue :',
        'enter_ticker': 'Entrez le ticker de l\'actif (ex: AAPL, TSLA, BTC-USD, CW8.PA):',
        'or_select_asset': 'Ou sélectionnez un actif dans la liste :',
        'chart_type': 'Type de graphique',
        'candlestick': 'Chandelier',
        'line': 'Ligne',
        'bar': 'Barre',
        'interval': 'Intervalle de temps des bougies :',
        'start_date': 'Date de début',
        'end_date': 'Date de fin',
        'use_all_data': 'Utiliser toutes les données disponibles',
        'technical_indicators': '📈 Indicateurs Techniques',
        'select_indicators': 'Sélectionnez les indicateurs à afficher :',
        'select_sma_periods': 'Périodes pour SMA (sélection multiple) :',
        'select_ema_periods': 'Périodes pour EMA (sélection multiple) :',
        'rsi_period': 'Période pour RSI :',
        'download_data': 'Télécharger les Données',
        'download_csv': 'Télécharger les données en CSV',
        'price': 'Prix',
        'date': 'Date',
        'no_data': '❌ Aucune donnée disponible pour le ticker saisi. Veuillez vérifier le ticker et réessayer.',
        'chart_title': 'Cours de ',
        'annotations': 'Annotations',
        'add_annotation': 'Ajouter une annotation',
        'annotation_text': 'Texte de l\'annotation',
        'annotation_date': 'Date de l\'annotation',
        'annotation_price': 'Prix de l\'annotation',
        'select_color': 'Sélectionnez la couleur :',
        'chart_customization': '🎨 Personnalisation du Graphique',
        'select_colors': 'Personnalisez les couleurs :',
        'main_color': 'Couleur principale',
        'background_color': 'Couleur de fond',
        'grid_color': 'Couleur de la grille',
        'prediction_model': '🔮 Modèle de Prédiction',
        'select_model': 'Sélectionnez le modèle de prédiction :',
        'arima': 'ARIMA',
        'lstm': 'LSTM',
        'no_model': 'Aucun',
        'model_parameters': 'Paramètres du Modèle LSTM',
        'evaluate_model': 'Évaluer le modèle',
        'mse': 'Erreur Quadratique Moyenne (MSE) :',
        'mae': 'Erreur Absolue Moyenne (MAE) :',
        'zoom_yaxis': 'Zoom sur l\'axe des ordonnées (Prix)',
        'yaxis_range': 'Définir la plage des prix (Y-axis) :',
        'min_price': 'Prix minimum',
        'max_price': 'Prix maximum',
        'forecasting': 'Prévision',
        'periods_to_predict': 'Nombre de périodes à prédire dans le futur :',
        'optimizing_arima': 'Optimisation des paramètres ARIMA...',
        'selected_arima_order': 'Ordre ARIMA sélectionné :',
        'training_lstm': 'Entraînement du modèle LSTM...',
        'error_downloading_data': 'Erreur lors du téléchargement des données :',
        'error_in_arima': 'Erreur dans le modèle ARIMA :',
    },
    'English': {
        'title': '📈 Enhanced Financial Markets Dashboard',
        'language_selection': 'Select language:',
        'enter_ticker': 'Enter the asset ticker (e.g., AAPL, TSLA, BTC-USD, CW8.PA):',
        'or_select_asset': 'Or select an asset from the list:',
        'chart_type': 'Chart Type',
        'candlestick': 'Candlestick',
        'line': 'Line',
        'bar': 'Bar',
        'interval': 'Time interval:',
        'start_date': 'Start Date',
        'end_date': 'End Date',
        'use_all_data': 'Use all available data',
        'technical_indicators': '📈 Technical Indicators',
        'select_indicators': 'Select indicators to display:',
        'select_sma_periods': 'Periods for SMA (multiple selection):',
        'select_ema_periods': 'Periods for EMA (multiple selection):',
        'rsi_period': 'Period for RSI:',
        'download_data': 'Download Data',
        'download_csv': 'Download data as CSV',
        'price': 'Price',
        'date': 'Date',
        'no_data': '❌ No data available for the entered ticker. Please check the ticker and try again.',
        'chart_title': 'Price of ',
        'annotations': 'Annotations',
        'add_annotation': 'Add an annotation',
        'annotation_text': 'Annotation text',
        'annotation_date': 'Annotation date',
        'annotation_price': 'Annotation price',
        'select_color': 'Select color:',
        'chart_customization': '🎨 Chart Customization',
        'select_colors': 'Customize colors:',
        'main_color': 'Main color',
        'background_color': 'Background color',
        'grid_color': 'Grid color',
        'prediction_model': '🔮 Prediction Model',
        'select_model': 'Select prediction model:',
        'arima': 'ARIMA',
        'lstm': 'LSTM',
        'no_model': 'None',
        'model_parameters': 'LSTM Model Parameters',
        'evaluate_model': 'Evaluate Model',
        'mse': 'Mean Squared Error (MSE):',
        'mae': 'Mean Absolute Error (MAE):',
        'zoom_yaxis': 'Zoom on Y-axis (Price)',
        'yaxis_range': 'Set Price Range (Y-axis):',
        'min_price': 'Minimum Price',
        'max_price': 'Maximum Price',
        'forecasting': 'Forecasting',
        'periods_to_predict': 'Number of periods to predict into the future:',
        'optimizing_arima': 'Optimizing ARIMA parameters...',
        'selected_arima_order': 'Selected ARIMA order:',
        'training_lstm': 'Training LSTM model...',
        'error_downloading_data': 'Error downloading data:',
        'error_in_arima': 'Error in ARIMA model:',
    }
}

# Initialiser les variables de session
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_unscaled' not in st.session_state:
    st.session_state.predictions_unscaled = None
if 'actuals_unscaled' not in st.session_state:
    st.session_state.actuals_unscaled = None
if 'mse' not in st.session_state:
    st.session_state.mse = None
if 'mae' not in st.session_state:
    st.session_state.mae = None
if 'future_predictions' not in st.session_state:
    st.session_state.future_predictions = None
if 'future_dates' not in st.session_state:
    st.session_state.future_dates = None

# Sélection de la langue
language = st.sidebar.selectbox(languages['Français']['language_selection'], ['Français', 'English'])
lang = languages[language]

# Titre de l'application
st.title(lang['title'])

# ==============================================
# Barre Latérale : Sélection de l'Actif
# ==============================================
st.sidebar.header(lang['enter_ticker'])
ticker_input = st.sidebar.text_input(lang['enter_ticker'], value="AAPL")

# Liste prédéfinie d'actifs
st.sidebar.markdown(lang['or_select_asset'])
assets = {
    'Actions': {
        'Apple Inc. (AAPL)': 'AAPL',
        'Microsoft Corporation (MSFT)': 'MSFT',
        'Tesla, Inc. (TSLA)': 'TSLA',
        'Amazon.com, Inc. (AMZN)': 'AMZN',
        'Alphabet Inc. (GOOGL)': 'GOOGL',
        'Meta Platforms, Inc. (META)': 'META'
    },
    'Indices': {
        'S&P 500 (^GSPC)': '^GSPC',
        'Dow Jones (^DJI)': '^DJI',
        'NASDAQ Composite (^IXIC)': '^IXIC',
        'FTSE 100 (^FTSE)': '^FTSE',
        'Nikkei 225 (^N225)': '^N225'
    },
    'ETFs': {
        'Amundi MSCI World UCITS ETF (CW8.PA)': 'CW8.PA',
        'Lyxor PEA Monde (MSCI World) UCITS ETF (WPEA.PA)': 'WPEA.PA'
    },
    'Cryptocurrencies': {
        'Bitcoin (BTC-USD)': 'BTC-USD',
        'Ethereum (ETH-USD)': 'ETH-USD',
        'Dogecoin (DOGE-USD)': 'DOGE-USD',
        'Cardano (ADA-USD)': 'ADA-USD',
        'Solana (SOL-USD)': 'SOL-USD'
    },
    'Currencies': {
        'EUR/USD (EURUSD=X)': 'EURUSD=X',
        'USD/JPY (JPY=X)': 'JPY=X',
        'GBP/USD (GBPUSD=X)': 'GBPUSD=X',
        'AUD/USD (AUDUSD=X)': 'AUDUSD=X',
        'USD/CAD (USDCAD=X)': 'USDCAD=X'
    }
}

asset_category = st.sidebar.selectbox("Sélectionnez une catégorie d'actifs / Select asset category:", list(assets.keys()))
asset_name = st.sidebar.selectbox("Sélectionnez un actif / Select an asset:", list(assets[asset_category].keys()))
selected_ticker = assets[asset_category][asset_name]

# Utiliser le ticker sélectionné si aucun ticker manuel n'est saisi
if not ticker_input or ticker_input.strip() == '':
    ticker_input = selected_ticker

# ==============================================
# Barre Latérale : Personnalisation du Graphique
# ==============================================
st.sidebar.header(lang['chart_customization'])

# Type de graphique
chart_type = st.sidebar.selectbox(lang['chart_type'], [lang['candlestick'], lang['line'], lang['bar']])

# Personnalisation des couleurs
st.sidebar.subheader(lang['select_colors'])
main_color = st.sidebar.color_picker(lang['main_color'], '#1f77b4')
background_color = st.sidebar.color_picker(lang['background_color'], '#111111')
grid_color = st.sidebar.color_picker(lang['grid_color'], '#333333')

# Intervalle de temps des bougies
interval_options = ['1h', '1d', '1wk', '1mo']
interval_label = lang['interval']
intervalle = st.sidebar.selectbox(interval_label, interval_options)

# Sélection de la plage de dates
st.sidebar.subheader(lang['use_all_data'])
use_all_data = st.sidebar.checkbox(lang['use_all_data'])

if use_all_data:
    date_fin = datetime.now()
    if intervalle == '1h':
        date_debut = date_fin - timedelta(days=729)  # Limite de 729 jours pour rester en dessous de 730
    else:
        date_debut = None  # yfinance accepte None pour récupérer toutes les données disponibles
else:
    if intervalle == '1h':
        default_start_date = (datetime.now() - timedelta(days=729)).date()
        date_debut_input = st.sidebar.date_input(lang['start_date'], value=default_start_date)
        min_start_date = (datetime.now() - timedelta(days=729)).date()
        # Convertir les dates en datetime.datetime pour la comparaison
        date_debut = datetime.combine(date_debut_input, datetime.min.time())
        max_start_date = datetime.combine(min_start_date, datetime.min.time())
        if date_debut < max_start_date:
            st.sidebar.warning(f"Pour l'intervalle 1h, la date de début ne peut pas être antérieure à {min_start_date}.")
            date_debut = max_start_date
    else:
        date_debut_input = st.sidebar.date_input(lang['start_date'], value=(datetime.now().date() - timedelta(days=365)))
        date_debut = datetime.combine(date_debut_input, datetime.min.time())
    date_fin_input = st.sidebar.date_input(lang['end_date'], value=datetime.now().date())
    date_fin = datetime.combine(date_fin_input, datetime.min.time())

    # Vérifier que la plage de dates est valide
    if date_debut and date_fin and date_debut > date_fin:
        st.sidebar.error("La date de début doit être antérieure à la date de fin.")
        st.stop()

# ==============================================
# Barre Latérale : Indicateurs Techniques
# ==============================================
st.sidebar.header(lang['technical_indicators'])
indicateurs = st.sidebar.multiselect(lang['select_indicators'], ['SMA', 'EMA', 'RSI', 'MACD', 'Bandes de Bollinger'])

# Sélection multiple des périodes pour SMA et EMA
sma_periods = []
ema_periods = []
periode_RSI = 14  # Valeur par défaut pour la période RSI
selected_SMA = None

if 'SMA' in indicateurs:
    sma_periods = st.sidebar.multiselect(lang['select_sma_periods'], [5, 10, 20, 50, 100, 200], default=[20])
    if sma_periods:
        selected_SMA = sma_periods[0]  # Sélectionner le premier SMA pour le modèle

if 'EMA' in indicateurs:
    ema_periods = st.sidebar.multiselect(lang['select_ema_periods'], [5, 10, 20, 50, 100, 200], default=[20])

if 'RSI' in indicateurs:
    periode_RSI = st.sidebar.number_input(lang['rsi_period'], min_value=1, value=14)

# ==============================================
# Barre Latérale : Modèle de Prédiction
# ==============================================
st.sidebar.header(lang['prediction_model'])
model_choice = st.sidebar.selectbox(lang['select_model'], [lang['no_model'], 'LSTM'])

# Prévision
st.sidebar.subheader(lang['forecasting'])
periods_to_predict = st.sidebar.number_input(lang['periods_to_predict'], min_value=1, max_value=365, value=30)

# ==============================================
# Barre Latérale : Paramètres du Modèle LSTM
# ==============================================
if model_choice == 'LSTM':
    st.sidebar.header(lang['model_parameters'])
    epochs = st.sidebar.number_input('Époques (Epochs)', min_value=1, max_value=100, value=20, step=1)
    batch_size = st.sidebar.number_input('Taille de lot (Batch Size)', min_value=16, max_value=512, value=32, step=16)
    seq_length = st.sidebar.number_input('Longueur de séquence (Sequence Length)', min_value=10, max_value=200, value=60, step=10)

# ==============================================
# Fonction de Téléchargement des Données
# ==============================================
@st.cache_data
def telecharger_donnees(ticker, start, end, intervalle):
    try:
        data = yf.download(ticker, start=start, end=end, interval=intervalle)
        if data.empty:
            raise ValueError(f"Aucune donnée trouvée pour le ticker {ticker} avec l'intervalle {intervalle}.")
        return data
    except Exception as e:
        st.error(f"{lang['error_downloading_data']} {e}")
        return pd.DataFrame()

# Gestion des dates en fonction de l'intervalle
if intervalle == '1h' and use_all_data:
    data = telecharger_donnees(ticker_input, date_debut, date_fin, intervalle)
elif not use_all_data and intervalle == '1h':
    # Pour '1h' avec dates manuelles, assurez-vous que date_debut >= date_fin - 729 jours
    max_start_date = date_fin - timedelta(days=729)
    if date_debut < max_start_date:
        st.sidebar.warning(f"Pour l'intervalle 1h, la date de début ne peut pas être antérieure à {max_start_date.date()}.")
        date_debut = max_start_date
    data = telecharger_donnees(ticker_input, date_debut, date_fin, intervalle)
else:
    data = telecharger_donnees(ticker_input, date_debut, date_fin, intervalle)

# Afficher les dates utilisées pour le téléchargement (pour débogage)
st.write(f"**Date de début pour téléchargement :** {date_debut}")
st.write(f"**Date de fin pour téléchargement :** {date_fin}")
st.write(f"**Intervalle :** {intervalle}")

# Vérification des données
if data.empty:
    st.error(lang['no_data'])
    st.stop()
else:
    actif_selectionne = ticker_input.upper()

    # ==============================================
    # Fonction de Calcul des Indicateurs Techniques
    # ==============================================
    @st.cache_data
    def calculer_indicateurs(data, indicateurs, sma_periods, ema_periods, periode_RSI):
        data = data.copy()  # Éviter de modifier les données originales

        if 'SMA' in indicateurs:
            for period in sma_periods:
                data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        if 'EMA' in indicateurs:
            for period in ema_periods:
                data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
        if 'RSI' in indicateurs:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=periode_RSI).rsi()
        if 'MACD' in indicateurs:
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Hist'] = macd.macd_diff()
        if 'Bandes de Bollinger' in indicateurs:
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_High'] = bollinger.bollinger_hband()
            data['BB_Low'] = bollinger.bollinger_lband()
        return data

    data = calculer_indicateurs(data, indicateurs, sma_periods, ema_periods, periode_RSI)

    # ==============================================
    # Préparation des Séquences pour le Modèle LSTM
    # ==============================================

    if model_choice == 'LSTM':
        # Sélection des caractéristiques pour le modèle
        feature_columns = ['Open', 'Close', 'High', 'Low']
        if selected_SMA:
            feature_columns.append(f'SMA_{selected_SMA}')
        # Assurez-vous que la SMA sélectionnée existe dans les données
        model_data = data[feature_columns].dropna()

        # Mise à l'échelle des données
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(model_data)

        # Création des séquences
        def create_sequences(data, seq_length):
            X = []
            Y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                Y.append(data[i + seq_length])
            return np.array(X), np.array(Y)

        X, Y = create_sequences(scaled_data, seq_length)

        # Définir les tailles des ensembles
        total_samples = X.shape[0]
        train_size = int(0.8 * total_samples)
        valid_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - valid_size

        X_train = X[:train_size]
        Y_train = Y[:train_size]

        X_valid = X[train_size:train_size + valid_size]
        Y_valid = Y[train_size:train_size + valid_size]

        X_test = X[train_size + valid_size:]
        Y_test = Y[train_size + valid_size:]

        # Conversion en tenseurs PyTorch
        X_train_tensor = torch.tensor(X_train).float()
        Y_train_tensor = torch.tensor(Y_train).float()

        X_valid_tensor = torch.tensor(X_valid).float()
        Y_valid_tensor = torch.tensor(Y_valid).float()

        X_test_tensor = torch.tensor(X_test).float()
        Y_test_tensor = torch.tensor(Y_test).float()

        # Création des datasets et dataloaders
        batch_size = batch_size  # Utiliser la valeur définie dans la sidebar

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ==============================================
        # Définition du Modèle LSTM avec PyTorch
        # ==============================================
        class NeuralNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(NeuralNetwork, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # Vérifiez si x a une seule séquence (pas de "batch"), modifiez h0 et c0 en conséquence
                if len(x.shape) == 2:  # si x est (seq_length, input_size) -> convertir en (1, seq_length, input_size)
                    x = x.unsqueeze(0)

                # Initialisation des états cachés avec les bonnes dimensions
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # h0: (num_layers, batch_size, hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # c0: (num_layers, batch_size, hidden_size)

                out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
                out = self.fc(out[:, -1, :])    # Prendre la sortie de la dernière étape de temps
                return out

        input_size = X_train.shape[2]  # Nombre de caractéristiques
        hidden_size = 64
        num_layers = 2
        output_size = Y_train.shape[1]  # Nombre de caractéristiques à prédire

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralNetwork(input_size, hidden_size, num_layers, output_size).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # ==============================================
        # Fonctions d'Entraînement et d'Évaluation
        # ==============================================
        def train_model(model, dataloader, optimizer, criterion):
            model.train()
            epoch_loss = 0
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            return epoch_loss / len(dataloader.dataset)

        def evaluate_model(model, dataloader, criterion):
            model.eval()
            epoch_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in dataloader:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, Y_batch)
                    epoch_loss += loss.item() * X_batch.size(0)
            return epoch_loss / len(dataloader.dataset)

        # ==============================================
        # Entraînement du Modèle
        # ==============================================
        if model_choice == 'LSTM':
            if st.sidebar.button(lang['evaluate_model']):
                with st.spinner('Entraînement du modèle LSTM en cours...'):
                    epochs_to_run = epochs
                    for epoch in range(1, epochs_to_run + 1):
                        train_loss = train_model(model, train_dataloader, optimizer, criterion)
                        valid_loss = evaluate_model(model, valid_dataloader, criterion)
                        st.write(f"Époque {epoch}/{epochs_to_run} - Perte d'entraînement: {train_loss:.4f} - Perte de validation: {valid_loss:.4f}")

                # Marquer que le modèle a été entraîné
                st.session_state.model_trained = True

                # ==============================================
                # Prédictions sur l'Ensemble de Test
                # ==============================================
                model.eval()
                predictions = []
                actuals = []
                with torch.no_grad():
                    for X_batch, Y_batch in test_dataloader:
                        X_batch = X_batch.to(device)
                        Y_batch = Y_batch.to(device)
                        preds = model(X_batch)
                        predictions.append(preds.cpu().numpy())
                        actuals.append(Y_batch.cpu().numpy())

                predictions = np.concatenate(predictions, axis=0)
                actuals = np.concatenate(actuals, axis=0)

                # Inversion de la mise à l'échelle
                predictions_unscaled = scaler.inverse_transform(predictions)
                actuals_unscaled = scaler.inverse_transform(actuals)

                # Stocker les résultats dans la session
                st.session_state.predictions_unscaled = predictions_unscaled
                st.session_state.actuals_unscaled = actuals_unscaled

                # ==============================================
                # Calcul des Métriques de Performance
                # ==============================================
                mse = mean_squared_error(actuals_unscaled, predictions_unscaled)
                mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)

                st.session_state.mse = mse
                st.session_state.mae = mae

                # Génération des Prédictions Futures
                with st.spinner('Génération des prédictions futures...'):
                    future_predictions = []
                    future_dates = []
                    last_sequence = scaled_data[-seq_length:].tolist()  # Récupération de la dernière séquence d'entrée

                    # Vérifiez si last_sequence est bien une liste de listes (et non un seul élément)
                    if not isinstance(last_sequence[0], list):
                        last_sequence = [last_sequence]

                    # Générer les prédictions futures pour le nombre de périodes spécifiées
                    for _ in range(periods_to_predict):
                        input_seq = torch.tensor(last_sequence).float().to(device)  # Pas besoin de doubler les crochets ici
                        with torch.no_grad():
                            # Prédiction avec le modèle
                            future_pred = model(input_seq).cpu().numpy()[0]
                        
                        # Ajouter la prédiction à la liste des futures prédictions
                        future_predictions.append(future_pred)
                        
                        # Mettre à jour la séquence en ajoutant la nouvelle prédiction et en supprimant la première valeur
                        last_sequence = last_sequence[1:] + [future_pred.tolist()]  # Assurez-vous que future_pred est converti en liste

                    # Inverser la mise à l'échelle des prédictions futures
                    future_predictions_unscaled = scaler.inverse_transform(future_predictions)
                    
                    # Générer les dates futures à partir de la dernière date dans les données
                    last_date = data.index[-1]
                    if intervalle == '1h':
                        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=periods_to_predict, freq='H')
                    elif intervalle == '1d':
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods_to_predict, freq='D')
                    elif intervalle == '1wk':
                        future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=periods_to_predict, freq='W')
                    elif intervalle == '1mo':
                        future_dates = pd.date_range(start=last_date + timedelta(days=30), periods=periods_to_predict, freq='M')
                    
                    # Stocker les prédictions futures et les dates dans l'état de la session
                    st.session_state.future_predictions = future_predictions_unscaled
                    st.session_state.future_dates = future_dates


    # ==============================================
    # Affichage des Prédictions et Métriques
    # ==============================================
    if st.session_state.model_trained:
        if st.session_state.predictions_unscaled is not None and st.session_state.actuals_unscaled is not None:
            mse = st.session_state.mse
            mae = st.session_state.mae

            st.subheader(f"{lang['mse']} {mse:.4f}")
            st.subheader(f"{lang['mae']} {mae:.4f}")

            # ==============================================
            # Affichage des Prédictions vs Réel
            # ==============================================
            test_dates = data.index[-len(st.session_state.actuals_unscaled):]

            # Vérifier si 'Close' est bien dans feature_columns
            close_index = feature_columns.index('Close') if 'Close' in feature_columns else 1
            st.write(f"Index de la colonne 'Close' utilisée pour les prédictions : {close_index}")

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=test_dates,
                y=st.session_state.actuals_unscaled[:, close_index],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            fig_pred.add_trace(go.Scatter(
                x=test_dates,
                y=st.session_state.predictions_unscaled[:, close_index],
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            fig_pred.update_layout(
                title=f"LSTM Predictions vs Actual Prices for {ticker_input}",
                xaxis_title=lang['date'],
                yaxis_title=lang['price'],
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font_color='white',
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color)
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # ==============================================
            # Affichage des Prédictions Futures (ajouté)
            # ==============================================
            if st.session_state.future_predictions is not None and st.session_state.future_dates is not None:
                fig_future = go.Figure()

                # Tracer les prédictions actuelles
                fig_future.add_trace(go.Scatter(
                    x=test_dates,
                    y=st.session_state.actuals_unscaled[:, close_index],
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))

                fig_future.add_trace(go.Scatter(
                    x=test_dates,
                    y=st.session_state.predictions_unscaled[:, close_index],
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red')
                ))

                # Tracer les prédictions futures
                fig_future.add_trace(go.Scatter(
                    x=st.session_state.future_dates,
                    y=st.session_state.future_predictions[:, close_index],
                    mode='lines',
                    name='Forecasted',
                    line=dict(color='green', dash='dash')
                ))

                fig_future.update_layout(
                    title=f"Prévision LSTM pour {ticker_input}",
                    xaxis_title=lang['date'],
                    yaxis_title=lang['price'],
                    plot_bgcolor=background_color,
                    paper_bgcolor=background_color,
                    font_color='white',
                    xaxis=dict(gridcolor=grid_color),
                    yaxis=dict(gridcolor=grid_color)
                )
                st.plotly_chart(fig_future, use_container_width=True)


    # ==============================================
    # Création du Graphique Principal
    # ==============================================
    fig = go.Figure()

    # Type de graphique
    if chart_type == lang['candlestick']:
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=actif_selectionne
        ))
    elif chart_type == lang['line']:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=actif_selectionne,
            line=dict(color=main_color)
        ))
    elif chart_type == lang['bar']:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Close'],
            name=actif_selectionne,
            marker_color=main_color
        ))

    # Ajout des Indicateurs Techniques
    if 'SMA' in indicateurs:
        for period in sma_periods:
            if f'SMA_{period}' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[f'SMA_{period}'],
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(width=2)
                ))
    if 'EMA' in indicateurs:
        for period in ema_periods:
            if f'EMA_{period}' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[f'EMA_{period}'],
                    mode='lines',
                    name=f'EMA {period}',
                    line=dict(width=2, dash='dash')
                ))
    if 'Bandes de Bollinger' in indicateurs:
        if 'BB_High' in data.columns and 'BB_Low' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_High'],
                mode='lines',
                line=dict(color='rgba(173,216,230,0.2)'),
                name='Bollinger High'
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Low'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(173,216,230,0.2)',
                line=dict(color='rgba(173,216,230,0.2)'),
                name='Bollinger Low'
            ))

    # Personnalisation du Graphique Principal
    fig.update_layout(
        title=lang['chart_title'] + actif_selectionne,
        xaxis_title=lang['date'],
        yaxis_title=lang['price'],
        xaxis_rangeslider_visible=False,
        height=600,
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font_color='white',
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )

    # Zoom sur l'axe Y
    st.sidebar.header(lang['zoom_yaxis'])
    if st.sidebar.checkbox(lang['yaxis_range']):
        min_price = st.sidebar.number_input(lang['min_price'], value=float(data['Close'].min()))
        max_price = st.sidebar.number_input(lang['max_price'], value=float(data['Close'].max()))
        fig.update_yaxes(range=[min_price, max_price])

    # ==============================================
    # Ajouter des Annotations
    # ==============================================
    st.sidebar.header(lang['annotations'])
    if st.sidebar.checkbox(lang['add_annotation']):
        annotation_text = st.sidebar.text_input(lang['annotation_text'], '')
        annotation_date_input = st.sidebar.date_input(lang['annotation_date'], value=datetime.now().date())
        annotation_date = datetime.combine(annotation_date_input, datetime.min.time())
        annotation_price = st.sidebar.number_input(lang['annotation_price'], value=float(data['Close'].mean()))
        annotation_color = st.sidebar.color_picker(lang['select_color'], '#FFFFFF')

        # Ajouter l'annotation au graphique
        fig.add_annotation(
            x=annotation_date,
            y=annotation_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color=annotation_color),
            arrowcolor=annotation_color
        )

    # Afficher le graphique principal
    st.plotly_chart(fig, use_container_width=True)

    # ==============================================
    # Afficher les Indicateurs Supplémentaires (MACD)
    # ==============================================
    if 'MACD' in indicateurs:
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                line=dict(color='blue'),
                name='MACD'
            ))
            fig_macd.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                line=dict(color='orange'),
                name='Signal'
            ))
            fig_macd.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Hist'],
                name='MACD Histogram'
            ))
            fig_macd.update_layout(
                title='MACD',
                height=300,
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font_color='white',
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color)
            )
            st.plotly_chart(fig_macd, use_container_width=True)

    st.write("Future Predictions:", st.session_state.future_predictions)
    st.write("Future Dates:", st.session_state.future_dates)

    # ==============================================
    # Option pour Télécharger les Données en CSV
    # ==============================================
    st.sidebar.header(lang['download_data'])

    @st.cache_data
    def convertir_csv(df):
        return df.to_csv().encode('utf-8')

    csv_data = convertir_csv(data)

    st.sidebar.download_button(
        label=lang['download_csv'],
        data=csv_data,
        file_name=f'{actif_selectionne}_data.csv',
        mime='text/csv',
    )
