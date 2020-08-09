## M5 Forecasting (Accuracy) Competition Hosted on Kaggle

The goal of this competition is to forecast 28 days ahead unit sales of 3,049 products (classified in 3 product categories: Hobbies, Foods, and Household) sold across 10 Walmarts stores located in CA, TX, and WI.

### Background

The Makridakis Competitions (also known as the M Competitions or M-Competitions) are a series of open competitions organized by teams led by forecasting researcher Spyros Makridakis and intended to evaluate and compare the accuracy of different forecasting methods.

The M5 Competition consists of two parallel tracks using the same dataset, the first requiring 28 days ahead point forecasts (M5 Forecasting - Accuracy) and the second 28 days ahead probabilistic forecasts for the median and four prediction intervals: 50%, 67%, 95%, and 99% (M5 Forecasting - Uncertainty).

Here are some helpful links:

* https://en.wikipedia.org/wiki/Makridakis_Competitions
* https://www.kaggle.com/c/m5-forecasting-accuracy/
* https://mofc.unic.ac.cy/m5-competition/

### Getting Started

Install prerequisites:
```
$ conda env create -f environment.yml 
```

Setup ```kaggle``` CLI tool: https://www.kaggle.com/docs/api

Run the following commands to download and decompress rquired data:
```
$ conda activate m5
$ cd data/input
$ kaggle download -c m5-forecasting-accuracy
$ unzip m5-forecasting-accuracy.zip
```

Run the pipeline:
```
$ TBD
```
### Acknowledgements

### References
