@echo off
:: Set the Python path to include the current directory
echo Setting up environment...
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Start the Streamlit application
echo Starting StockVision AI Dashboard...
python -m streamlit run app.py

:: Wait for a moment to ensure the application starts
timeout /t 5 /nobreak

:: Open the default web browser with the application URL
echo Opening browser...
start http://localhost:8501

:: Keep the window open
echo Application is running. Press Ctrl+C to stop.
pause
