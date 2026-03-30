API Reference
============

REST API Endpoints
------------------

Health Check
~~~~~~~~~~~

.. http:get:: /health

   Check API health status.

   **Example Request:**

   .. code-block:: bash

       curl http://localhost:8000/health

   **Example Response:**

   .. json::

       {
         "status": "healthy",
         "version": "3.0.0"
       }

Prediction
~~~~~~~~~~

.. http:post:: /predict

   Get stock prediction.

   **Request Body:**

   .. json::

       {
         "symbol": "AAPL",
         "model": "ensemble",
         "days": 30
       }

   **Example Request:**

   .. code-block:: bash

       curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"symbol": "AAPL"}'

   **Example Response:**

   .. json::

       {
         "symbol": "AAPL",
         "current_price": 150.00,
         "predictions": [
           {"date": "2024-01-15", "price": 152.50, "lower": 148.00, "upper": 157.00}
         ],
         "model": "ensemble",
         "timestamp": "2024-01-01T12:00:00Z"
       }

Models
~~~~~~

.. http:get:: /models/info

   Get available models.

   **Example Response:**

   .. json::

       {
         "models": [
           {"name": "arima", "version": "1.0.0"},
           {"name": "lstm", "version": "1.0.0"},
           {"name": "ensemble", "version": "1.0.0"}
         ]
       }

Python SDK
----------

You can also use Pythia as a Python library:

.. code-block:: python

    from src.models.predictor import StockPredictor

    predictor = StockPredictor()
    predictor.load_data('AAPL')
    predictor.train()
    prediction = predictor.predict(days=30)

Data Collection
---------------

.. code-block:: python

    from src.data.collector import StockDataCollector

    collector = StockDataCollector()
    data = collector.download_yahoo_data('AAPL')

Fundamental Analysis
--------------------

.. code-block:: python

    from src.analytics.fundamentals import FundamentalAnalyzer

    analyzer = FundamentalAnalyzer()
    analysis = analyzer.get_full_analysis('AAPL')

Trading Signals
---------------

.. code-block:: python

    from src.ml.signals import SignalGenerator

    generator = SignalGenerator()
    signals = generator.generate_signals('AAPL')
