import yfinance as yf

sector_tickers = [
    'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI',
    'XLP', 'XLU', 'XLB', 'XLRE', 'XLC'
]

sector_tickers = ['SPY']

data = yf.download(
    sector_tickers,
    start='2017-01-01',
    end='2021-12-31',
    group_by='ticker',
    auto_adjust=True
)

data.columns = ['_'.join(col) for col in data.columns]
data.to_csv('data/prices.csv')

# Example: get adjusted close prices
# adj_close = data.xs('Close', level=1, axis=1)
# print(adj_close.head())
