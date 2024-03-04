import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selectedStock = st.selectbox("Select dataset for prediction", stocks)

nYears = st.slider("Years of prediction:", 1, 4)
period = nYears * 365

@st.cache_data                                                                            # with this once a specific data is downloaded like for example APPL
def loadData(ticker):                                                                     # it's cached. This means that this fuction does not need to be run again 
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    
    return data

dataLoadState = st.text("Load Data.....")
data = loadData(selectedStock)
dataLoadState.text("Loading data.....done")

# analysing the data

st.subheader('Raw data')
st.write(data.tail())

def plotRawData():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plotRawData()

#Prediction

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})                        # This how prophet wants the data

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Prediction data')
st.write(forecast.tail())

st.write("Prediction data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Prediction components")
fig2 = m.plot_components(forecast)
st.write(fig2)