# RoadSight Intelligence

RoadSight Intelligence is an integrated Streamlit application for road, wildlife, audio, sign-language, person-analysis, and drowsiness intelligence demos.

## Features

- Car colour and traffic object detection
- Wildlife risk detection and species mapping
- Voice emotion detection
- Person attribute and emotion analysis
- Sign language detection
- Driver drowsiness detection

## Requirements

- Windows
- Python 3.11
- Internet access for downloading model weights or packages when required

## Setup

Open PowerShell in the project directory:

```powershell
cd "C:\Users\Lenovo\Documents\AI-ML-Internship-Project"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sign-language dependencies can be installed separately when needed:

```powershell
python -m pip install -r requirements-sign-language.txt
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

Press `Ctrl+C` in the terminal to stop the application.

## Tests

Run the test suite from the project directory:

```powershell
python -m pytest
```

## Model and Media Files

Large model weights and generated media are not included in Git. Some modules require documented local model files or will report an explicit missing-model error. See each module directory for its implementation and setup notes.

Do not add secrets, private data, generated media, virtual environments, or large model weights to the repository.