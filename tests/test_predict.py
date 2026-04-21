from src.predict import predict_text

def test_prediction():
    result = predict_text("I will hurt you")
    assert "harmful" in result
    assert result["harmful"] == 1 or result["harmful"] == 0
    assert "confidence" in result
