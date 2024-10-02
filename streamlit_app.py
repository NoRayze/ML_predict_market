import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import ta
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pmdarima import auto_arima
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
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
        'model_parameters': 'Paramètres du Modèle',
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
        'trading_strategy_parameters': 'Paramètres de la Stratégie de Trading',
        'trading_frequency': 'Fréquence de Trading',
        'transaction_costs': 'Frais de Transaction',
        'type_of_transaction_fee': 'Type de Frais de Transaction',
        'fixed': 'Fixe',
        'percentage': 'Pourcentage',
        'fixed_transaction_fee': 'Frais de Transaction Fixe (€)',
        'transaction_fee_percentage': 'Frais de Transaction (%)',
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
        'model_parameters': 'Model Parameters',
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
        'trading_strategy_parameters': 'Trading Strategy Parameters',
        'trading_frequency': 'Trading Frequency',
        'transaction_costs': 'Transaction Costs',
        'type_of_transaction_fee': 'Type of Transaction Fee',
        'fixed': 'Fixed',
        'percentage': 'Percentage',
        'fixed_transaction_fee': 'Fixed Transaction Fee (€)',
        'transaction_fee_percentage': 'Transaction Fee (%)',
    }
}

# Sélection de la langue
language = st.sidebar.selectbox("Sélectionnez la langue / Select language:", ['Français', 'English'])
lang = languages[language]

# Titre de l'application
st.title(lang['title'])

# ==============================================
# Barre latérale pour la sélection
# ==============================================

# Champ de saisie pour le ticker
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

asset_category = st.sidebar.selectbox("", list(assets.keys()))
asset_name = st.sidebar.selectbox("", list(assets[asset_category].keys()))
selected_ticker = assets[asset_category][asset_name]

# Si aucun ticker n'est saisi manuellement, utiliser le ticker sélectionné
if not ticker_input or ticker_input.strip() == '':
    ticker_input = selected_ticker

# Sélection du type de graphique
st.sidebar.header(lang['chart_customization'])
chart_type = st.sidebar.selectbox(lang['chart_type'], [lang['candlestick'], lang['line'], lang['bar']])

# Personnalisation des couleurs
st.sidebar.subheader(lang['select_colors'])
main_color = st.sidebar.color_picker(lang['main_color'], '#1f77b4')
background_color = st.sidebar.color_picker(lang['background_color'], '#111111')
grid_color = st.sidebar.color_picker(lang['grid_color'], '#333333')

# Sélection de l'intervalle de temps
intervalle = st.sidebar.selectbox(lang['interval'], ['1h', '1d', '1wk', '1mo'])

# Sélection de la plage de dates
st.sidebar.subheader(lang['use_all_data'])
use_all_data = st.sidebar.checkbox(lang['use_all_data'])

if use_all_data:
    date_debut = None  # yfinance accepte None pour récupérer toutes les données disponibles
    date_fin = datetime.now()
else:
    date_debut = st.sidebar.date_input(lang['start_date'], value=datetime.now() - timedelta(days=365))
    date_fin = st.sidebar.date_input(lang['end_date'], value=datetime.now())

# Sélection des indicateurs techniques
st.sidebar.header(lang['technical_indicators'])
indicateurs = st.sidebar.multiselect(lang['select_indicators'], ['SMA', 'EMA', 'RSI', 'MACD', 'Bandes de Bollinger'])

# Sélection multiple des périodes pour SMA et EMA
sma_periods = []
ema_periods = []
periode_RSI = 14  # Valeur par défaut pour la période RSI
if 'SMA' in indicateurs:
    sma_periods = st.sidebar.multiselect(lang['select_sma_periods'], [5, 10, 20, 50, 100, 200], default=[20])
if 'EMA' in indicateurs:
    ema_periods = st.sidebar.multiselect(lang['select_ema_periods'], [5, 10, 20, 50, 100, 200], default=[20])
if 'RSI' in indicateurs:
    periode_RSI = st.sidebar.number_input(lang['rsi_period'], min_value=1, value=14)

# ==============================================
# Optimisation des performances
# ==============================================

@st.cache_data
def telecharger_donnees(ticker, start, end, intervalle):
    try:
        data = yf.download(ticker, start=start, end=end, interval=intervalle)
        data = data.asfreq('B')  # Fréquence business day
        data = data.fillna(method='ffill')  # Remplir les valeurs manquantes
        return data
    except Exception as e:
        st.error(f"{lang['error_downloading_data']} {e}")
        return pd.DataFrame()

data = telecharger_donnees(ticker_input, date_debut, date_fin, intervalle)

# Vérifier si les données sont disponibles
if data.empty:
    st.error(lang['no_data'])
    st.stop()
else:
    actif_selectionne = ticker_input.upper()

    # Calcul des indicateurs techniques
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
    # Sélection du modèle de prédiction
    # ==============================================
    st.sidebar.header(lang['prediction_model'])
    model_choice = st.sidebar.selectbox(lang['select_model'], [lang['no_model'], 'ARIMA', 'LSTM'])

    # Préparer les données pour les modèles
    model_data = data[['Close']].dropna()

    # Entrée pour le nombre de périodes futures à prédire
    st.sidebar.subheader(lang['forecasting'])
    periods_to_predict = st.sidebar.number_input(lang['periods_to_predict'], min_value=1, max_value=365, value=30)

    # Ajouter des entrées pour les paramètres de backtesting
    st.sidebar.subheader("Backtesting Parameters")
    train_window_size = st.sidebar.number_input("Training Window Size (days)", min_value=30, value=200)
    prediction_window_size = st.sidebar.number_input("Prediction Window Size (days)", min_value=1, value=20)
    step_size = st.sidebar.number_input("Step Size (days)", min_value=1, value=20)

    # Ajouter la sélection de la temporalité des décisions
    st.sidebar.subheader(lang['trading_strategy_parameters'])
    trading_frequency = st.sidebar.selectbox(lang['trading_frequency'], options=["Daily", "Weekly", "Monthly"])

    # Ajouter les frais de transaction
    st.sidebar.subheader(lang['transaction_costs'])
    transaction_fee_type = st.sidebar.selectbox(lang['type_of_transaction_fee'], options=[lang['fixed'], lang['percentage']])
    if transaction_fee_type == lang['fixed']:
        transaction_fee = st.sidebar.number_input(lang['fixed_transaction_fee'], min_value=0.0, value=1.0, step=0.1)
    else:
        transaction_fee = st.sidebar.number_input(lang['transaction_fee_percentage'], min_value=0.0, max_value=100.0, value=0.1, step=0.1) / 100

    # Vérifier la taille des données
    if train_window_size >= len(model_data['Close']):
        st.error("La taille de la fenêtre d'entraînement est trop grande pour la quantité de données disponible.")
        st.stop()

    # Création du graphique principal
    fig = go.Figure()

    # Sélection du type de graphique
    if chart_type == lang['candlestick']:
        fig.add_trace(go.Candlestick(
            x=model_data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=model_data['Close'],
            name=actif_selectionne
        ))
    elif chart_type == lang['line']:
        fig.add_trace(go.Scatter(
            x=model_data.index,
            y=model_data['Close'],
            mode='lines',
            name=actif_selectionne,
            line=dict(color=main_color)
        ))
    elif chart_type == lang['bar']:
        fig.add_trace(go.Bar(
            x=model_data.index,
            y=model_data['Close'],
            name=actif_selectionne,
            marker_color=main_color
        ))

    # Ajouter les indicateurs techniques
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

    # ==============================================
    # Modèles de prédiction avec Backtesting
    # ==============================================

    if model_choice == 'ARIMA':
        st.sidebar.subheader(lang['model_parameters'])
        st.write("Running ARIMA Backtesting...")

        # Convertir la fréquence de trading en nombre de jours
        if trading_frequency == "Daily":
            trading_step = 1
        elif trading_frequency == "Weekly":
            trading_step = 5  # Environ 5 jours ouvrables par semaine
        elif trading_frequency == "Monthly":
            trading_step = 21  # Environ 21 jours ouvrables par mois

        # Fonction de backtesting pour ARIMA
        def arima_backtesting(data, train_window_size, prediction_window_size, step_size):
            predictions = []
            actuals = []
            dates = []
            start_index = 0
            end_index = train_window_size
            iteration = 0  # Compteur d'itérations

            # Initialiser le portefeuille
            initial_capital = 1000.0
            capital = initial_capital
            positions = 0  # Nombre d'unités de l'actif détenues
            portfolio_values = []  # Valeur du portefeuille au fil du temps

            while end_index + prediction_window_size <= len(data):
                st.write(f"Iteration {iteration}: start_index={start_index}, end_index={end_index}")

                train_data = data.iloc[start_index:end_index]
                test_data = data.iloc[end_index:end_index + prediction_window_size]

                if len(train_data) == 0 or len(test_data) == 0:
                    st.write("Taille des données insuffisante pour cette itération.")
                    break

                try:
                    # Entraîner le modèle sur train_data
                    model_arima = auto_arima(train_data['Close'], start_p=1, start_q=1,
                                             max_p=5, max_q=5, seasonal=False,
                                             d=None, trace=False,
                                             error_action='ignore', suppress_warnings=True, stepwise=True,
                                             max_d=2,
                                             information_criterion='aic')

                    model_arima_fit = model_arima.fit(train_data['Close'])

                    # Prédire sur test_data
                    forecast = model_arima_fit.predict(n_periods=len(test_data))
                    predictions.extend(forecast)
                    actuals.extend(test_data['Close'].values)
                    dates.extend(test_data.index)

                    # Simuler la stratégie de trading
                    for i in range(0, len(test_data), trading_step):
                        if i + trading_step > len(test_data):
                            break

                        # Prix actuel et prédiction
                        current_price = test_data['Close'].iloc[i]
                        predicted_price = forecast[i]

                        # Décision de trading
                        if predicted_price > current_price and positions == 0:
                            # Calcul des frais de transaction
                            if transaction_fee_type == lang['fixed']:
                                fee = transaction_fee
                            else:
                                fee = transaction_fee * capital

                            # Vérifier si le capital est suffisant pour couvrir les frais
                            if capital <= fee:
                                st.write(f"Fonds insuffisants pour couvrir les frais à {test_data.index[i]}")
                                continue

                            # Acheter l'actif
                            capital_after_fee = capital - fee
                            positions = capital_after_fee / current_price
                            capital = 0
                            st.write(f"Achat à {test_data.index[i]} au prix de {current_price:.2f}, frais de {fee:.2f}€")
                        elif predicted_price < current_price and positions > 0:
                            # Vendre l'actif
                            proceeds = positions * current_price
                            # Calcul des frais de transaction
                            if transaction_fee_type == lang['fixed']:
                                fee = transaction_fee
                            else:
                                fee = transaction_fee * proceeds

                            capital = proceeds - fee
                            positions = 0
                            st.write(f"Vente à {test_data.index[i]} au prix de {current_price:.2f}, frais de {fee:.2f}€")

                        # Calcul de la valeur du portefeuille
                        portfolio_value = capital + positions * current_price
                        portfolio_values.append({
                            'Date': test_data.index[i],
                            'Portfolio Value': portfolio_value
                        })

                except Exception as e:
                    st.write(f"Erreur lors de l'entraînement ou de la prédiction : {e}")
                    break

                # Avancer la fenêtre
                start_index += step_size
                end_index += step_size
                iteration += 1

            # Vérifier si des prédictions ont été générées
            if len(predictions) == 0:
                st.error("Aucune prédiction n'a été générée. Veuillez vérifier les paramètres du backtesting.")
                return pd.DataFrame(), None, None, None

            # Créer un DataFrame avec les résultats
            results_df = pd.DataFrame({
                'Date': dates,
                'Actual': actuals,
                'Predicted': predictions
            }).set_index('Date')

            # Créer un DataFrame pour la valeur du portefeuille
            portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')

            # Calculer les métriques de performance
            mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
            mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

            # Calcul du rendement final
            final_portfolio_value = capital + positions * data['Close'].iloc[-1]
            total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100

            st.write(f"Valeur finale du portefeuille : {final_portfolio_value:.2f}€")
            st.write(f"Rendement total de la stratégie : {total_return:.2f}%")

            return results_df, mse, mae, portfolio_df

        # Exécuter le backtesting
        results_df, mse, mae, portfolio_df = arima_backtesting(model_data, train_window_size, prediction_window_size, step_size)

        # Vérifier si des résultats ont été générés
        if results_df.empty or portfolio_df is None:
            st.stop()

        st.write(f"{lang['mse']} {mse:.4f}")
        st.write(f"{lang['mae']} {mae:.4f}")

        # Tracer les résultats du backtesting
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Actual'],
            mode='lines',
            name='Actual'
        ))
        fig_backtest.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Predicted'],
            mode='lines',
            name='Predicted'
        ))
        st.plotly_chart(fig_backtest, use_container_width=True)

        # Tracer la valeur du portefeuille au fil du temps
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['Portfolio Value'],
            mode='lines',
            name='Portfolio Value'
        ))
        st.plotly_chart(fig_portfolio, use_container_width=True)

    elif model_choice == 'LSTM':
        st.sidebar.subheader(lang['model_parameters'])
        # Paramètres d'entraînement personnalisables
        epochs = st.sidebar.number_input('Epochs', min_value=1, max_value=50, value=5, step=1)
        batch_size = st.sidebar.number_input('Batch Size', min_value=16, max_value=256, value=64, step=16)
        seq_length = st.sidebar.number_input('Sequence Length', min_value=10, max_value=200, value=50, step=10)

        st.write("Running LSTM Backtesting...")

        # Convertir la fréquence de trading en nombre de jours
        if trading_frequency == "Daily":
            trading_step = 1
        elif trading_frequency == "Weekly":
            trading_step = 5  # Environ 5 jours ouvrables par semaine
        elif trading_frequency == "Monthly":
            trading_step = 21  # Environ 21 jours ouvrables par mois

        # Fonction de backtesting pour LSTM
        def create_sequences(data, seq_length):
            X = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
            return np.array(X)

        def lstm_backtesting(data, seq_length, train_window_size, prediction_window_size, step_size):
            predictions = []
            actuals = []
            dates = []
            start_index = 0
            end_index = train_window_size
            iteration = 0  # Compteur d'itérations

            # Initialiser le portefeuille
            initial_capital = 1000.0
            capital = initial_capital
            positions = 0  # Nombre d'unités de l'actif détenues
            portfolio_values = []  # Valeur du portefeuille au fil du temps

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

            while end_index + prediction_window_size <= len(scaled_data):
                st.write(f"Iteration {iteration}: start_index={start_index}, end_index={end_index}")

                train_data = scaled_data[start_index:end_index]

                if len(train_data) < seq_length:
                    st.write("Taille des données d'entraînement insuffisante pour la séquence.")
                    break

                X_train = create_sequences(train_data, seq_length)
                Y_train = train_data[seq_length:]

                # Reshape
                X_train = X_train.reshape((X_train.shape[0], seq_length, 1))
                Y_train = Y_train.reshape(-1, 1)

                # Construire le modèle
                model_lstm = tf.keras.models.Sequential([
                    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                    tf.keras.layers.LSTM(50),
                    tf.keras.layers.Dense(1)
                ])
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')

                # Entraîner le modèle
                model_lstm.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Prédire
                test_data = scaled_data[end_index - seq_length:end_index + prediction_window_size]
                X_test = create_sequences(test_data, seq_length)
                X_test = X_test.reshape((X_test.shape[0], seq_length, 1))

                forecast = model_lstm.predict(X_test)
                forecast = scaler.inverse_transform(forecast)

                actual = data.values[end_index:end_index + prediction_window_size]

                # Limiter la longueur si nécessaire
                min_length = min(len(forecast), len(actual))
                predictions.extend(forecast[:min_length].flatten())
                actuals.extend(actual[:min_length])
                dates.extend(data.index[end_index:end_index + min_length])

                # Simuler la stratégie de trading
                for i in range(0, min_length, trading_step):
                    if i + trading_step > min_length:
                        break

                    # Prix actuel et prédiction
                    current_price = actual[i]
                    predicted_price = forecast[i][0]

                    # Décision de trading
                    if predicted_price > current_price and positions == 0:
                        # Calcul des frais de transaction
                        if transaction_fee_type == lang['fixed']:
                            fee = transaction_fee
                        else:
                            fee = transaction_fee * capital

                        if capital <= fee:
                            st.write(f"Fonds insuffisants pour couvrir les frais à {data.index[end_index + i]}")
                            continue

                        # Acheter l'actif
                        capital_after_fee = capital - fee
                        positions = capital_after_fee / current_price
                        capital = 0
                        st.write(f"Achat à {data.index[end_index + i]} au prix de {current_price:.2f}, frais de {fee:.2f}€")
                    elif predicted_price < current_price and positions > 0:
                        # Vendre l'actif
                        proceeds = positions * current_price
                        if transaction_fee_type == lang['fixed']:
                            fee = transaction_fee
                        else:
                            fee = transaction_fee * proceeds

                        capital = proceeds - fee
                        positions = 0
                        st.write(f"Vente à {data.index[end_index + i]} au prix de {current_price:.2f}, frais de {fee:.2f}€")

                    # Calcul de la valeur du portefeuille
                    portfolio_value = capital + positions * current_price
                    portfolio_values.append({
                        'Date': data.index[end_index + i],
                        'Portfolio Value': portfolio_value
                    })

                # Avancer la fenêtre
                start_index += step_size
                end_index += step_size
                iteration += 1

            # Vérifier si des prédictions ont été générées
            if len(predictions) == 0:
                st.error("Aucune prédiction n'a été générée. Veuillez vérifier les paramètres du backtesting.")
                return pd.DataFrame(), None, None, None

            # Créer un DataFrame avec les résultats
            results_df = pd.DataFrame({
                'Date': dates,
                'Actual': actuals,
                'Predicted': predictions
            }).set_index('Date')

            # Créer un DataFrame pour la valeur du portefeuille
            portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')

            # Calculer les métriques de performance
            mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
            mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

            # Calcul du rendement final
            final_portfolio_value = capital + positions * data.values[-1]
            total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100

            st.write(f"Valeur finale du portefeuille : {final_portfolio_value:.2f}€")
            st.write(f"Rendement total de la stratégie : {total_return:.2f}%")

            return results_df, mse, mae, portfolio_df

        # Exécuter le backtesting
        results_df, mse, mae, portfolio_df = lstm_backtesting(model_data['Close'], seq_length, train_window_size, prediction_window_size, step_size)

        # Vérifier si des résultats ont été générés
        if results_df.empty or portfolio_df is None:
            st.stop()

        st.write(f"{lang['mse']} {mse:.4f}")
        st.write(f"{lang['mae']} {mae:.4f}")

        # Tracer les résultats du backtesting
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Actual'],
            mode='lines',
            name='Actual'
        ))
        fig_backtest.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Predicted'],
            mode='lines',
            name='Predicted'
        ))
        st.plotly_chart(fig_backtest, use_container_width=True)

        # Tracer la valeur du portefeuille au fil du temps
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['Portfolio Value'],
            mode='lines',
            name='Portfolio Value'
        ))
        st.plotly_chart(fig_portfolio, use_container_width=True)

    # ==============================================
    # Personnaliser les couleurs et le thème du graphique
    # ==============================================
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
        min_price = st.sidebar.number_input(lang['min_price'], value=float(model_data['Close'].min()))
        max_price = st.sidebar.number_input(lang['max_price'], value=float(model_data['Close'].max()))
        fig.update_yaxes(range=[min_price, max_price])

    # ==============================================
    # Ajouter des annotations
    # ==============================================
    st.sidebar.header(lang['annotations'])
    if st.sidebar.checkbox(lang['add_annotation']):
        annotation_text = st.sidebar.text_input(lang['annotation_text'], '')
        annotation_date = st.sidebar.date_input(lang['annotation_date'], value=model_data.index[len(model_data)//2])
        annotation_price = st.sidebar.number_input(lang['annotation_price'], value=float(model_data['Close'].mean()))
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

    # Afficher les indicateurs supplémentaires (par exemple, MACD)
    if 'MACD' in indicateurs:
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns and 'MACD_Hist' in data.columns:
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

    # Option pour télécharger les données en CSV
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
