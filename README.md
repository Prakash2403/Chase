# Automatic Stock trading bot (WIP)

Inspired by [Gekko](https://github.com/askmike/gekko)

## Overview

Aim of this project is to create a bot which can create a portfolio from a given list of stocks, optimizes the portfolio
by efficiently dividing the assets and then does automatic trading to maximize the profit.

Most tutorials on Internet(and my previous project on same topic) use only historial stock price 
to predict future prices. However in this project, important features are extracted from the data and are used in conjuction 
with historical prices. Due to this, it's performance is significantly better than other projects available on Internet.

## Target Audience

**Traders who know Finance but not Machine Learning or Programming**:

**Engineers who know Machine Learning but not Finance**:

**Software Developers**: Well, it took me some time to design the UML for this project. I tried to stick to 
**[S.O.L.I.D.](https://medium.com/@cramirez92/s-o-l-i-d-the-first-5-priciples-of-object-oriented-design-with-javascript-790f6ac9b9fa)** 
principles as much as possible. Although I am not an expert in this area, so if you think that there is scope for improvement, 
feel free to open an issue. 

## Current Status

Future price predictor and portfolio optimizer are ready independently. For predicting future price, following features have
been used
  
    1. Adjusted Close (Directly available from dataset)
    2. Volume (Directly available from dataset)
    3. Simple moving average (Computed explicitly)
    4. Daily Returns (Computed explicitly)
 
