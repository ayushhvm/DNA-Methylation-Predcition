# DNA Methylation Prediction App

A Streamlit web application for predicting DNA methylation states using sequence features.

## Features

- Upload and analyze DNA sequence data
- Train machine learning models for methylation prediction
- Visualize sequence features and predictions
- Make predictions on new DNA sequences

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Data Format

The application expects two CSV files:
- `train.csv`: Training data with DNA sequences and methylation scores
- `test.csv`: Test data with DNA sequences for prediction

Each file should have the following columns:
- `sequence`: DNA sequence (A, T, G, C)
- `methylation_score`: Methylation score (0-1) for training data

## Usage

1. Upload your training and test data
2. Explore the data statistics
3. Train a machine learning model
4. Make predictions on new sequences

## License

MIT License
