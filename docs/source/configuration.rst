Configuration
=============

Configuration File
-----------------

Pythia uses a ``config.yaml`` file for configuration. Here's a complete example:

.. code-block:: yaml

    data:
      source: yahoo
      cache_enabled: true
      cache_ttl: 300

    models:
      arima:
        order: [5, 1, 2]
        seasonal_order: [1, 1, 1, 12]
      lstm:
        units: 128
        layers: 2
        dropout: 0.2

    indicators:
      sma: [20, 50, 200]
      ema: [12, 26]
      rsi:
        period: 14
        overbought: 70
        oversold: 30
      macd:
        fast: 12
        slow: 26
        signal: 9

    sentiment:
      enabled: true
      sources: [news, twitter]

    api:
      host: 0.0.0.0
      port: 8000
      workers: 4
      rate_limit: 100

    logging:
      level: INFO
      file: logs/pythia.log

Environment Variables
--------------------

You can also configure Pythia using environment variables:

- ``PYTHIA_DATA_SOURCE``: Data source (yahoo, alphavantage)
- ``PYTHIA_CACHE_ENABLED``: Enable caching (true/false)
- ``PYTHIA_LOG_LEVEL``: Logging level
- ``API_HOST``: API server host
- ``API_PORT``: API server port
- ``REDIS_URL``: Redis connection URL

Presets
-------

Pythia includes preset configurations for different use cases:

Aggressive (High Risk/High Return)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    models:
      default: lstm
      weights:
        lstm: 0.6
        gru: 0.4

    indicators:
      short_ma: [10, 20]
      long_ma: [50, 100]

Conservative (Low Risk)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    models:
      default: arima
      weights:
        arima: 0.7
        lstm: 0.3

    indicators:
      short_ma: [50, 100]
      long_ma: [100, 200]

Balanced
~~~~~~~~

.. code-block:: yaml

    models:
      default: ensemble
      weights:
        arima: 0.33
        lstm: 0.33
        gru: 0.34

    indicators:
      sma: [20, 50, 200]
