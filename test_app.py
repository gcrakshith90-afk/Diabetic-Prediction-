import unittest
import json
from app import app

class FlaskAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Diabetes Prediction System", response.data)

    def test_prediction_endpoint_valid_json(self):
        payload = {
            "pregnancies": 6,
            "glucose": 148,
            "bloodPressure": 72,
            "skinThickness": 35,
            "insulin": 0,  # Should be imputed
            "bmi": 33.6,
            "diabetesPedigreeFunction": 0.627,
            "age": 50
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('prediction', data)
        self.assertIn('probability', data)
        self.assertIn('risk_level', data)
        self.assertIn(data['prediction'], [0, 1])
        self.assertGreaterEqual(data['probability'], 0)
        self.assertLessEqual(data['probability'], 100)
        print("\nTest prediction response:", data)

    def test_prediction_missing_field(self):
        payload = {
            "pregnancies": 2,
            "glucose": 100
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
