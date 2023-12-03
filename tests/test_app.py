import requests
import json
import pytest

# Base URL of your API
base_api_url = "http://127.0.0.1:5000/predict/"

def test_predict_digit_svm():
    return _test_predict_digit(base_api_url + 'svm')

def test_predict_digit_lr():
    return _test_predict_digit(base_api_url + 'lr')

def test_predict_digit_tree():
    return _test_predict_digit(base_api_url + 'tree')

def _test_predict_digit(api_url):
    # Define the input data as a Python list with 64 features
    input_data = [
        0.0, 0.0, 0.0, 11.999999999999982, 13.000000000000004, 5.000000000000021,
        8.881784197001265e-15, 0.0, 0.0, 0.0, 0.0, 10.999999999999986, 15.999999999999988,
        9.000000000000005, 1.598721155460224e-14, 0.0, 0.0, 0.0, 2.9999999999999925,
        14.999999999999979, 15.999999999999998, 6.000000000000022, 1.0658141036401509e-14,
        0.0, 6.217248937900871e-15, 6.999999999999987, 14.99999999999998, 15.999999999999996,
        16.0, 2.0000000000000284, 3.552713678800507e-15, 0.0, 5.5220263365470826e-30,
        6.21724893790087e-15, 1.0000000000000113, 15.99999999999998, 16.0, 3.000000000000022,
        5.32907051820075e-15, 0.0, 0.0, 0.0, 0.9999999999999989, 15.99999999999998, 16.0,
        6.000000000000015, 1.0658141036401498e-14, 0.0, 0.0, 0.0, 0.9999999999999989,
        15.99999999999998, 16.0, 6.000000000000018, 1.0658141036401503e-14, 0.0, 0.0, 0.0,
        0.0, 10.999999999999986, 15.999999999999993, 10.00000000000001, 1.7763568394002505e-14, 0.0
    ]

    # Create a dictionary payload from the input data
    payload = {"image": input_data}

    # Send a POST request to the API
    response = requests.post(api_url, json=payload)

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Parse the JSON response
    result = response.json()

    # Assert the expected prediction result
    # Note: The assert condition here is an example. You may need to modify it based on the actual expected output of your models.
    assert "y_predicted" in result
