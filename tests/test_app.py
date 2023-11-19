import pytest
from app import app

@pytest.mark.parametrize("input_data, expected_digit", [
    ([0.1, 0.2, 0.3, 0.4, 0.5], 0),
    ([0.5, 0.4, 0.3, 0.2, 0.1], 1),
    # Add more test cases for other digits here
])
def test_predict(input_data, expected_digit):
    client = app.test_client()
    response = client.post('/predict', json={"image": input_data})

    # Check if the response status code is 200 OK
    assert response.status_code == 200
    
    # Check if the predicted digit matches the expected digit
    assert int(response.get_json()["y_predicted"]) == expected_digit

# Add more test cases for other digits and status code here

