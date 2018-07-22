# Automatic Stock trading bot (WIP)

Inspired by [Gekko](https://github.com/askmike/gekko)

## Overview

Aim of this project is to create a bot which can create a portfolio from a given list of stocks, optimizes the portfolio
by efficiently dividing the assets and then does automatic trading to maximize the profit.

Most tutorials on Internet(and my previous project on same topic) use only historial stock price 
to predict future prices. However in this project, important features are extracted from the data and are used in conjuction 
with historical prices. Due to this, it's performance is significantly better than other projects available on Internet.

## Salient features.

### Single point of control

Almost everything can be controlled by changing the parameters in `config.py`. No need to change anything in core code.

### Designed using S.O.L.I.D. principles

No tight coupling and proper encapsulations and abstractions ensure that code is flexible and easy to extend. 

For example, if you want to use existing LSTM neural network for predicting another time series(say electricity consumption series), then you only need to extend the `DataHandler` base class, define your own methods to extract and preprocess the data and you are good to go. 

## Target Audience

### Traders who know Finance but not Machine Learning or Programming: 

### Engineers who know Machine Learning but not Finance:

### Software Developers: 

Well, it took me some time to design the UML for this project. I tried to stick to 
**[S.O.L.I.D.](https://medium.com/@cramirez92/s-o-l-i-d-the-first-5-priciples-of-object-oriented-design-with-javascript-790f6ac9b9fa)** 
principles as much as possible. I hope it may help you in case you face any design issues. Although I am not an expert in this area, so if you think that there is scope for improvement, feel free to open an issue. 

## Current Status

Future price predictor and portfolio optimizer are ready independently. For predicting future price, following features have
been used
  
    1. Adjusted Close (Directly available from dataset)
    2. Volume (Directly available from dataset)
    3. Simple moving average (Computed explicitly)
    4. Daily Returns (Computed explicitly)
 
