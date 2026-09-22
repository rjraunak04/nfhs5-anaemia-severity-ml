# Deploy the aggregate Streamlit dashboard

This dashboard ships two clearly labeled modes: aggregate NFHS development results (default) and a synthetic engineering demo. It does not host the trained model or accept respondent records.

## Local verification

With Python 3.12, install `python -m pip install -e ".[dev]"`, run `python -m pytest -q` and `python -m ruff check src tests app.py`, then start `streamlit run app.py`. Check all four dashboard tabs, switch to the synthetic demo, and confirm that the locked test is labeled untouched.

## Streamlit Community Cloud

After the repository visibility and disclosure review, open [share.streamlit.io](https://share.streamlit.io/). If the repository is still private, first grant Streamlit Community Cloud access to this repository in the connected GitHub account. Choose **Create app**, select `rjraunak04/nfhs5-anaemia-severity-ml`, branch `main`, entry point `app.py`, and Python **3.12**. The root `requirements.txt` installs the `[app]` extra. Wait for the app to load and verify the headline, the four tabs and the synthetic switch. Add the resulting verified live URL to the README and repository homepage.

If a cloud build fails, inspect the Streamlit build logs and check the selected Python version and `requirements.txt`. Do not upload the local dataset, `runs/` directory, checkpoint files or any `.joblib` artifact to the hosting service. A public dashboard of aggregate development results must not be advertised as clinical prediction or final test performance.
