# AI Spam Message Detector

A small Machine Learning web application that predicts whether a text message is **spam** or **not spam**. It uses TF-IDF to convert text into numerical features and Logistic Regression to classify messages. The interface is built with Streamlit.

> **Educational note:** This repository includes a small sample dataset for demonstration. The model is a learning prototype, not a production spam filter. Its predictions and confidence scores can be wrong.

## Features

- Analyze a user-entered message
- Predict `spam` or `ham` (not spam)
- Display the model's estimated confidence
- Try built-in sample messages
- Train locally from the included CSV dataset
- Automated tests for spam, normal messages, and empty input

## Tech stack

- Python 3.10+
- Streamlit
- pandas
- scikit-learn (TF-IDF + Logistic Regression)
- pytest

## Project structure

```text
ai-spam-message-detector/
├── app.py
├── model.py
├── data/messages.csv
├── tests/test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally (Windows PowerShell)

1. Install Python 3.10 or later.
2. Clone the repository and open its folder in a terminal.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   streamlit run app.py
   ```

6. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Run tests

```powershell
pytest -q
```

## How the Machine Learning works

1. The CSV file contains labeled example messages (`spam` and `ham`).
2. `TfidfVectorizer` converts text into weighted word and phrase features.
3. `LogisticRegression` learns patterns from those features.
4. The trained pipeline predicts the class of a new message and returns its estimated class probability.

The model is trained locally when the app starts; no paid API key or cloud AI account is required.

## Possible improvements

- Train on a larger, carefully sourced public SMS spam dataset.
- Evaluate using a train/test split and precision, recall, F1-score, and confusion matrix.
- Add multilingual support and explainable token-level insights.
- Deploy the Streamlit app after testing and reviewing the dataset license.

## License

No license is currently included. Add an appropriate license before redistributing this project or its dataset.