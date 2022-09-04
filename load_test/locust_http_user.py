import json

from locust import constant, HttpUser, task
from constants import MODULE_DIR



class RemixerUser(HttpUser):
    """
    Simulated AWS Lambda User
    """

    wait_time = constant(1)
    headers = {"Content-type": "application/json"}
    payload = json.dumps({"genre": "jazz"})

    @task
    def predict(self):
        response = self.client.post("/", data=self.payload, headers=self.headers)
        pred = response.json()["pred"]
