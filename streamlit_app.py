"""Streamlit Cloud entry point — the research report lives in run/report_app.py.

Streamlit Community Cloud defaults the main file to `streamlit_app.py` at the repo
root, so this shim just runs the real app. (You can equivalently set the deploy
"Main file path" to `run/report_app.py`.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run"))
import report_app  # noqa: E402  (sets st.set_page_config on import)

report_app.main()
