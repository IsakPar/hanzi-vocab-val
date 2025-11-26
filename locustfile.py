"""
Load test for vocab-validator

Run with:
  locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to start the test
"""

from locust import HttpUser, task, between
import random


# Sample texts of varying lengths
SAMPLE_TEXTS = [
    "你好！",  # Very short
    "你好，谢谢，再见。",  # Short
    "我学习中文。我很喜欢。",  # Medium
    "今天我和我的朋友去吃饭。我们吃了很多好吃的东西。",  # Longer
    "我每天早上六点起床。然后我吃早饭，喝咖啡。八点我去上班。",  # Paragraph
]

# Different user positions to test
USER_POSITIONS = [
    {"hsk": 1, "lesson": 1},
    {"hsk": 1, "lesson": 3},
    {"hsk": 1, "lesson": 5},
    {"hsk": 2, "lesson": 1},
    {"hsk": 2, "lesson": 3},
]


class ValidatorUser(HttpUser):
    """Simulates a backend service calling the validator"""
    
    # Wait 0.1-0.5 seconds between requests (simulates realistic usage)
    wait_time = between(0.1, 0.5)
    
    @task(10)  # Most common task
    def validate_text(self):
        """POST /validate - the main endpoint"""
        self.client.post("/validate", json={
            "text": random.choice(SAMPLE_TEXTS),
            "user_position": random.choice(USER_POSITIONS),
            "target_words": []
        })
    
    @task(2)  # Less common
    def health_check(self):
        """GET /health"""
        self.client.get("/health")
    
    @task(1)  # Rare
    def get_version(self):
        """GET /version"""
        self.client.get("/version")


class HeavyUser(HttpUser):
    """Simulates burst traffic - no wait between requests"""
    
    wait_time = between(0, 0.1)  # Almost no wait
    
    @task
    def validate_rapid(self):
        """Rapid-fire validation requests"""
        self.client.post("/validate", json={
            "text": "今天我和我的朋友去吃饭。我们吃了很多好吃的东西。我很开心。",
            "user_position": {"hsk": 1, "lesson": 5},
            "target_words": ["吃饭", "朋友"]
        })

