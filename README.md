\# OrbitShieldAI



\## Overview



OrbitShieldAI is a Python-based research prototype for satellite conjunction screening. It combines a Streamlit dashboard, TLE-based orbital screening, and a machine learning classifier trained on historical data.



The project is intended for educational and research purposes.



\## Features



\- Streamlit dashboard for interacting with the application.

\- TLE-based screening of two satellites over a selected time window.

\- Closest-approach estimation using sampled orbital positions.

\- Configurable separation threshold assessment.

\- Simulated maneuver recommendation for research purposes.

\- Saved machine learning classifier trained on historical data.

\- Automated tests for basic application functions and report parsing.



\## Technology Stack



\- Python

\- Streamlit

\- scikit-learn

\- Joblib

\- pytest

\- TLE-based orbital data



\## Running the Application



1\. Open a terminal in the project directory.

2\. Activate the project environment or use its Python executable.

3\. Run:



&#x20;  ```cmd

&#x20;  .venv313\\Scripts\\python.exe -m streamlit run app.py

&#x20;  ```



4\. Open the local URL displayed in the terminal.



\## Running Tests



Run the automated tests with:



```cmd

.venv313\\Scripts\\python.exe -m pytest -v

```



\## Important Limitations



\- This is a research prototype, not an operational collision-avoidance system.

\- Orbital screening estimates depend on available TLE data and finite time sampling.

\- A sampled minimum separation is not a certified collision probability.

\- The machine learning classifier is trained on historical data and is not integrated into the live orbital screening calculation.

\- Simulated maneuver recommendations are illustrative and must not be used to control real spacecraft.

\- Results should not be interpreted as proof that a satellite encounter is safe.



\## Project Status



The Streamlit dashboard, historical machine learning workflow, TLE-based screening, basic input validation, and initial automated tests have been implemented.



Further work includes documentation review, final demonstration, version control setup, GitHub publication, and deployment testing.

