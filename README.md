# M5 (Accuracy) Forecasting Competition Hosted on Kaggle

* https://en.wikipedia.org/wiki/Makridakis_Competitions
* https://www.kaggle.com/c/m5-forecasting-accuracy/

The following basic trading strategies are implemented and evaluated as part of this analysis:

* Bollinger Bands
* Simple Moving Average Cross Over
* Buy and Hold

**[Market-Data-EDA.ipynb](notebooks/Market-Data-EDA.ipynb)**

We start by [analysing](notebooks/Market-Data-EDA.ipynb) the [dataset downloaded from Kaggle]([1]) that contains stock and ETF OHLC (Open, High, Low, Close) prices. This dataset consists of a large number of individual files with each containing market data for one ticker. We first develop a market data API that abstracts away access to ticker files before proceeding with an EDA. We also showcase usage of [Dask]([2]) given that the market data is spread across a number of files and all of the data cannot fit into memory on my laptop.

**[Basic-Strategies.ipynb](notebooks/Basic-Strategies.ipynb)**

We then [implement a simple backtester](notebooks/Basic-Strategies.ipynb) for running trading strategies. As part of it we develop a number of Python classes: Strategy, Backtest, ConfigSearch, etc. The backtester allows us to search for the best perforing ticker & strategy combinations. We run a backtest for all 3 strategies using 5 year's worth of market data.

### Getting Started

Install prerequisites:
```
$ conda env create -f environment.yml 
```
Jupyter notebooks are located under ```notebooks``` directory:

* [Market-Data-EDA.ipynb](notebooks/Market-Data-EDA.ipynb)
* [Basic-Strategies.ipynb](notebooks/Basic-Strategies.ipynb)

### Acknowledgements

* The strategy backtesting notebook is inspired by the following Kaggle analysis:
  - https://www.kaggle.com/malyvsen/algotrading-analysis
* Some of the ideas around backtesting are inspired by [quantstart.com](https://www.quantstart.com/)
  - https://www.quantstart.com/articles/Should-You-Build-Your-Own-Backtester/
  - https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/

### Potential Future Improvements

* Implement and evaluate the Mean Reversion Pairs Trading strategy
  - https://www.quantstart.com/articles/Backtesting-An-Intraday-Mean-Reversion-Pairs-Strategy-Between-SPY-And-IWM/
  - https://www.pythonforfinance.net/2018/07/04/mean-reversion-pairs-trading-with-inclusion-of-a-kalman-filter/
  - https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a
  
* Develop a more sophisticated backtester

### References

* Kaggle dataset:
  * https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
* [Quantopian lectures](https://www.quantopian.com/lectures)

[1]: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
[2]: https://dask.org/
