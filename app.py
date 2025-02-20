import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from numba import jit

# ----------------------------
# Data Generation & Preprocessing
# ----------------------------

def generate_data():
    """
    Generate synthetic stock data for demonstration.
    In a real-world scenario, replace this with actual data loading (e.g., from Yahoo Finance).
    """
    dates = pd.date_range(start='2020-01-01', end='2020-12-31')
    stocks = ['StockA', 'StockB', 'StockC']
    data_frames = []

    for stock in stocks:
        # Simulate price and dividend data with a slight trend and some noise
        price = np.linspace(100, 150, len(dates)) + np.random.randn(len(dates)) * 5
        dividend = np.linspace(2, 3, len(dates)) + np.random.randn(len(dates)) * 0.2
        df = pd.DataFrame({
            'Date': dates,
            'Stock': stock,
            'Price': price,
            'Dividend': dividend
        })
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)

# Initial synthetic dataset
df = generate_data()

# ----------------------------
# Computation with Numba Acceleration
# ----------------------------

@jit(nopython=True)
def compute_growth(prices, dividends):
    """
    Compute the period-to-period growth rate of dividends.
    Uses numba for performance acceleration.
    """
    growth = np.empty(len(prices))
    growth[0] = 0.0  # No growth for the first entry
    for i in range(1, len(prices)):
        if dividends[i-1] != 0:
            growth[i] = (dividends[i] - dividends[i-1]) / dividends[i-1]
        else:
            growth[i] = 0.0
    return growth

def add_growth_rate(df):
    """
    Add a 'GrowthRate' column to the DataFrame for each stock.
    """
    results = []
    for stock in df['Stock'].unique():
        stock_df = df[df['Stock'] == stock].sort_values('Date').copy()
        growth = compute_growth(stock_df['Price'].values, stock_df['Dividend'].values)
        stock_df['GrowthRate'] = growth
        results.append(stock_df)
    return pd.concat(results, ignore_index=True)

# Enhance our DataFrame with the growth rate calculation
df = add_growth_rate(df)

# ----------------------------
# Building the Dash Dashboard
# ----------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dividend Growth Insights Dashboard"),

    html.Div([
        html.Label("Select Stock:"),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': s, 'value': s} for s in df['Stock'].unique()],
            value='StockA'
        )
    ], style={'width': '25%', 'display': 'inline-block'}),

    dcc.Graph(id='price-chart'),
    dcc.Graph(id='dividend-chart'),
    dcc.Graph(id='growth-chart')
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('dividend-chart', 'figure'),
     Output('growth-chart', 'figure')],
    [Input('stock-dropdown', 'value')]
)
def update_graphs(selected_stock):
    filtered = df[df['Stock'] == selected_stock]

    price_fig = {
        'data': [go.Scatter(
            x=filtered['Date'],
            y=filtered['Price'],
            mode='lines',
            name='Price'
        )],
        'layout': go.Layout(
            title='Stock Price Over Time',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'}
        )
    }

    dividend_fig = {
        'data': [go.Scatter(
            x=filtered['Date'],
            y=filtered['Dividend'],
            mode='lines',
            name='Dividend'
        )],
        'layout': go.Layout(
            title='Dividend Over Time',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Dividend'}
        )
    }

    growth_fig = {
        'data': [go.Scatter(
            x=filtered['Date'],
            y=filtered['GrowthRate'],
            mode='lines',
            name='Growth Rate'
        )],
        'layout': go.Layout(
            title='Dividend Growth Rate Over Time',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Growth Rate'}
        )
    }

    return price_fig, dividend_fig, growth_fig

if __name__ == '__main__':
    app.run_server(debug=True)
