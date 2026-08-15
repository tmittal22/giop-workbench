@echo off
REM Launch the GIOP Workbench in your browser.
cd /d "%~dp0"
where python >nul 2>nul || ( echo Install Python from python.org, ticking "Add to PATH". & pause & exit /b 1 )
if not exist "src\giop\data\optics_coef.txt" (
  echo Fetching the reference optical tables ^(one time, needs internet^)...
  python scripts\fetch_data.py || ( pause & exit /b 1 )
)
python -m pip install --quiet numpy scipy matplotlib streamlit
python -m streamlit run app/giop_app.py
pause
