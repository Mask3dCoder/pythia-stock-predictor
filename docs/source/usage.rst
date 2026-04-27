Usage
=====

Command Line Interface
----------------------

After installation, use the ``pythia`` command:

Basic Prediction
~~~~~~~~~~~~~~~~

.. code-block:: bash

    pythia --symbol AAPL --predict

With specific model:

.. code-block:: bash

    pythia --symbol AAPL --model ensemble --days 30

With technical indicators:

.. code-block:: bash

    pythia --symbol AAPL --indicators rsi,macd

Terminal Interface (TUI)
-----------------------

Start the interactive terminal:

.. code-block:: bash

    pythia-tui

Available Commands:

- ``Q <symbol>`` - Quote
- ``GP <symbol>`` - Graph/Chart
- ``AI <symbol>`` - AI Analysis
- ``IN <indicator> <symbol>`` - Technical Indicators
- ``OPT <symbol>`` - Options Chain
- ``NW <symbol>`` - News
- ``PORT`` - Portfolio
- ``RSK`` - Risk Metrics
- ``FIND <term>`` - Search
- ``WL`` - Watchlist
- ``PRED <symbol>`` - AI Prediction
- ``BT <symbol>`` - Backtest
- ``CR <symbol>`` - Crypto
- ``FX <pair>`` - Forex
- ``FUND <symbol>`` - Fundamental Analysis
- ``SECTOR`` - Sector Performance
- ``BREADTH`` - Market Breadth
- ``PATTERN <symbol>`` - Pattern Recognition
- ``ANOMALY <symbol>`` - Anomaly Detection
- ``SIGNAL <symbol>`` - Trading Signals

REST API
--------

Start the API server:

.. code-block:: bash

    pythia-api

Or using Docker:

.. code-block:: bash

    docker-compose up api

API Examples
~~~~~~~~~~~~

Get prediction:

.. code-block:: bash

    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"symbol": "AAPL", "days": 30}'

Check health:

.. code-block:: bash

    curl http://localhost:8000/health

Streamlit Dashboard
-------------------

Start the dashboard:

.. code-block:: bash

    pythia-dashboard

Or using Docker:

.. code-block:: bash

    docker-compose up streamlit

Python API
----------

Use Pythia in your Python code:

.. code-block:: python

    from src.models.predictor import StockPredictor

    predictor = StockPredictor('AAPL', 'ensemble')

    predictor.load_data(years=5)
    predictor.train()
    predictions = predictor.predict(steps=30)
    print(predictions['predictions'])

Fundamental Analysis:

.. code-block:: python

    from src.analytics.fundamentals import FundamentalAnalyzer

    analyzer = FundamentalAnalyzer()
    analysis = analyzer.get_full_analysis('AAPL')

Trading Signals:

.. code-block:: python

    from src.ml.signals import SignalGenerator

    generator = SignalGenerator()
    signals = generator.generate_signals('AAPL')
