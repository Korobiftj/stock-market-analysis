```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('C:/Users/fatem/Downloads/stocks.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAPL</td>
      <td>2023-02-07</td>
      <td>150.639999</td>
      <td>155.229996</td>
      <td>150.639999</td>
      <td>154.649994</td>
      <td>154.414230</td>
      <td>83322600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAPL</td>
      <td>2023-02-08</td>
      <td>153.880005</td>
      <td>154.580002</td>
      <td>151.169998</td>
      <td>151.919998</td>
      <td>151.688400</td>
      <td>64120100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAPL</td>
      <td>2023-02-09</td>
      <td>153.779999</td>
      <td>154.330002</td>
      <td>150.419998</td>
      <td>150.869995</td>
      <td>150.639999</td>
      <td>56007100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAPL</td>
      <td>2023-02-10</td>
      <td>149.460007</td>
      <td>151.339996</td>
      <td>149.220001</td>
      <td>151.009995</td>
      <td>151.009995</td>
      <td>57450700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAPL</td>
      <td>2023-02-13</td>
      <td>150.949997</td>
      <td>154.259995</td>
      <td>150.919998</td>
      <td>153.850006</td>
      <td>153.850006</td>
      <td>62199000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Date'] = pd.to_datetime(df['Date'])
```


```python
print(df.isnull().sum())
```

    Ticker       0
    Date         0
    Open         0
    High         0
    Low          0
    Close        0
    Adj Close    0
    Volume       0
    dtype: int64
    


```python
print(df.groupby('Ticker')['Close'].describe())
```

            count        mean        std         min         25%         50%  \
    Ticker                                                                     
    AAPL     62.0  158.240645   7.360485  145.309998  152.077499  158.055000   
    GOOG     62.0  100.631532   6.279464   89.349998   94.702501  102.759998   
    MSFT     62.0  275.039839  17.676231  246.270004  258.742500  275.810013   
    NFLX     62.0  327.614677  18.554419  292.760010  315.672493  325.600006   
    
                   75%         max  
    Ticker                          
    AAPL    165.162506  173.570007  
    GOOG    105.962503  109.459999  
    MSFT    287.217506  310.649994  
    NFLX    338.899994  366.829987  
    

## Exploratory Data Analysis (EDA)

### Price Trends over time


```python
plt.figure(figsize=(14,8))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker']==ticker]
    plt.plot(ticker_data['Date'],ticker_data['Close'],label=ticker)

plt.title('Stock Price Trends (FEB- MAY 2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_8_0.png)
    


### Calculate daily returns


```python

df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()

plt.figure(figsize=(14, 8))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    plt.plot(ticker_data['Date'], ticker_data['Daily_Return'], label=ticker)
    
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_10_0.png)
    


### Volatitlity analysis


```python
volatility = df.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility
print("\nAnnualized Volatility:")
print(volatility.sort_values(ascending=False))
```

    
    Annualized Volatility:
    Ticker
    NFLX    0.356880
    GOOG    0.328764
    MSFT    0.283849
    AAPL    0.224660
    Name: Daily_Return, dtype: float64
    

### Feature Engineering


```python
# Calculate 10-day and 30-day moving averages
df['MA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
df['MA_30'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())

# Plot moving averages for AAPL
aapl = df[df['Ticker'] == 'AAPL']
plt.figure(figsize=(14, 6))
plt.plot(aapl['Date'], aapl['Close'], label='AAPL Close Price')
plt.plot(aapl['Date'], aapl['MA_10'], label='10-day MA')
plt.plot(aapl['Date'], aapl['MA_30'], label='30-day MA')
plt.title('Apple Moving Averages')
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Calculate RSI (14-day)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

for ticker in df['Ticker'].unique():
    mask = df['Ticker'] == ticker
    df.loc[mask, 'RSI'] = compute_rsi(df[mask])
```

### Correlation Analysis


```python
# Pivot to get closing prices by date
pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')

# Calculate correlation matrix
corr_matrix = pivot_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Stock Price Correlations')
plt.show()
```


    
![png](output_17_0.png)
    


# Performance Comparison

### Cumulative Returns


```python
# Calculate cumulative returns
for ticker in df['Ticker'].unique():
    mask = df['Ticker'] == ticker
    df.loc[mask, 'Cumulative_Return'] = (1 + df.loc[mask, 'Daily_Return']).cumprod()

plt.figure(figsize=(14, 8))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    plt.plot(ticker_data['Date'], ticker_data['Cumulative_Return'], label=ticker)
    
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_20_0.png)
    


### Risk-Return Profile


```python
# Calculate average daily return and volatility
performance = df.groupby('Ticker').agg({
    'Daily_Return': ['mean', 'std'],
    'Close': ['first', 'last']
})

performance.columns = ['Avg_Daily_Return', 'Daily_Volatility', 'Initial_Price', 'Final_Price']
performance['Total_Return'] = (performance['Final_Price'] - performance['Initial_Price']) / performance['Initial_Price']
performance['Annualized_Volatility'] = performance['Daily_Volatility'] * np.sqrt(252)

print("\nPerformance Metrics:")
print(performance.sort_values('Total_Return', ascending=False))
```

    
    Performance Metrics:
            Avg_Daily_Return  Daily_Volatility  Initial_Price  Final_Price  \
    Ticker                                                                   
    MSFT            0.002606          0.017881     267.559998   310.649994   
    AAPL            0.001991          0.014152     154.649994   173.570007   
    GOOG           -0.000067          0.020710     108.040001   106.214996   
    NFLX           -0.001676          0.022481     362.950012   322.760010   
    
            Total_Return  Annualized_Volatility  
    Ticker                                       
    MSFT        0.161048               0.283849  
    AAPL        0.122341               0.224660  
    GOOG       -0.016892               0.328764  
    NFLX       -0.110732               0.356880  
    


```python

```
