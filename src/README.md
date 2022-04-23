# A deep learning approach to predict significant wave height using long short-term memory

The codes in this folder:

* lstm_v1.py

    --> Univariate forecast of Hs using only ERA5 historical data for training (features and target).

* lstm_v2.py

    --> Univariate forecast of Hs using only ERA5 historical data for training (features) and buoy data as target.

* lstm_multi.py

    --> Multivariate forecast of Hs using only ERA5 historical data for training (features and target). Wind speed, direction and peak period can be considered as features, or the best correlated variables with Hs.

* lstm_historic.py

    --> This code is the same as ``lstm_v1.py``, but iterating the starting dates for the training, from '1979-01-01' until '2017-01-01'.