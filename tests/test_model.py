import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from model import train_model, predict_message

@pytest.fixture(scope="module")
def model():
    return train_model()

def test_model_predicts_obvious_spam(model):
    result = predict_message(model, "Congratulations! Claim your free cash prize now")
    assert result["label"] == "spam"
    assert 0.0 <= result["confidence"] <= 1.0

def test_model_predicts_normal_message(model):
    result = predict_message(model, "Can we meet for lunch after class tomorrow?")
    assert result["label"] == "ham"
    assert 0.0 <= result["confidence"] <= 1.0

def test_empty_message_is_rejected(model):
    with pytest.raises(ValueError):
        predict_message(model, "   ")
