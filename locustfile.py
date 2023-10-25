import random
import time

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(5, 20)

    @task
    def ask_question(self):
        self.client.get("/")
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "history": [{"user": random.choice(["Provide a AW server description", "How many system components in a AW server?", "Can you install AW server on CT console", "How to install  AW Server Client on CT Console?"])}],
                "approach": "rrr",
                "overrides": {"retrieval_mode": "hybrid", "semantic_ranker": True, "semantic_captions": False, "top": 3, "suggest_followup_questions": False},
            },
        )
        time.sleep(5)
        self.client.post(
            "/chat",
            json={
                "history": [
                    {
                        "user": "Provide a AW server description?",
                        "bot": "AW Server is a software package delivered with off-the-shelf, server-class hardware that allows easy selection, review, processing and filming of multiple-modality DICOM images from a variety of PC client machines via LAN or WAN networks. It also allows the user to choose lossless or lossy compression schemes to make a trade-off between speed and quality. AW Server is intended to be used in a manner similar to the current GE Health Care AW workstation product. It will be used to create and review diagnostic evidence related to radiology procedures by trained physicians in General Purpose Radiology, Oncology, Cardiology and Neurology clinical areas. [AW Install Manual-3.pdf].",
                    },
                    {"user": "Which Other GE healthcare product it can be integrated?"},
                ],
                "approach": "rrr",
                "overrides": {"retrieval_mode": "hybrid", "semantic_ranker": True, "semantic_captions": False, "top": 3, "suggest_followup_questions": False},
            },
        )
