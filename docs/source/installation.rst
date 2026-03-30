Installation
============

Requirements
-----------

- Python 3.9+
- pip

Install from PyPI
----------------

.. code-block:: bash

    pip install pythia-stock-predictor

Install with ML dependencies:

.. code-block:: bash

    pip install pythia-stock-predictor[ml]

Install with visualization:

.. code-block:: bash

    pip install pythia-stock-predictor[visualization]

Install all dependencies:

.. code-block:: bash

    pip install pythia-stock-predictor[ml,visualization]

Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/Mask3dCoder/pythia-stock-predictor.git
    cd pythia-stock-predictor
    pip install -e .

Docker
------

Using Docker Compose:

.. code-block:: bash

    docker-compose up -d

This will start:

- API server on http://localhost:8000
- Redis cache on port 6379
- Streamlit dashboard on http://localhost:8501

Environment Variables
--------------------

- ``API_HOST``: API server host (default: 0.0.0.0)
- ``API_PORT``: API server port (default: 8000)
- ``LOG_LEVEL``: Logging level (default: info)
- ``CACHE_ENABLED``: Enable caching (default: true)
