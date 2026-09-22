# Deploy the aggregate Streamlit dashboard

This dashboard ships two clearly labeled modes: aggregate NFHS development results (default) and a synthetic engineering demo. It does not host the trained model or accept respondent records.

## Local verification

With Python 3.12, install `python -m pip install -e ".[dev]"`, run `python -m pytest -q` and `python -m ruff check src tests app.py`, then start `streamlit run app.py`. Check all four dashboard tabs, switch to the synthetic demo, and confirm that the locked test is labeled untouched.

## Streamlit Community Cloud

After the repository visibility and disclosure review, open [share.streamlit.io](https://share.streamlit.io/). If the repository is still private, first grant Streamlit Community Cloud access to this repository in the connected GitHub account. Choose **Create app**, select `rjraunak04/nfhs5-anaemia-severity-ml`, branch `main`, entry point `app.py`, and Python **3.12**. The root `requirements.txt` installs the `[app]` extra. Wait for the app to load and verify the headline, the four tabs and the synthetic switch. Add the resulting verified live URL to the README and repository homepage.

If a cloud build fails, inspect the Streamlit build logs and check the selected Python version and `requirements.txt`. Do not upload the local dataset, `runs/` directory, checkpoint files or any `.joblib` artifact to the hosting service. A public dashboard of aggregate development results must not be advertised as clinical prediction or final test performance.


## Railway

The repository is Railway-ready through the root `Dockerfile`. Railway automatically detects a root Dockerfile and injects a `PORT` environment variable for the running service. The container starts Streamlit on that port and uses the same port for its health check.

After connecting Railway to GitHub:

1. Create a new Railway project and choose **Deploy from GitHub repo**.
2. Select `rjraunak04/nfhs5-anaemia-severity-ml` and branch `main`.
3. Keep the detected root `Dockerfile` build.
4. Set the Railway deployment healthcheck path to `/_stcore/health`.
5. Generate a public Railway domain and open the dashboard.
6. Verify the disclosure banner, all four tabs, the synthetic-data switch, and that the locked test remains labeled untouched.
7. Add the verified live URL to the repository homepage and README only after this check.

Railway can auto-deploy future pushes from the connected GitHub branch. Do not add NFHS/DHS raw data, model binaries, local checkpoints, or secrets as deployment files or environment variables.


## Current production deployment

Verified Railway production URL: https://nfhs5-anaemia-severity-ml-production.up.railway.app

The latest production deployment completed successfully with the root Dockerfile, dynamic Railway `PORT` binding, and `/_stcore/health` configured as the service healthcheck.
