# File: utils/communication_tracker.py
from collections import defaultdict 
class CommunicationTracker:
    def __init__(self):
        self.total_upload = 0.0  # in GB
        self.total_download = 0.0  # in GB
        self.per_round = {}  # {round: {"upload": GB, "download": GB}}
        self.per_client = defaultdict(lambda: {"upload": 0.0, "download": 0.0})

    def log_round(self, server_round, upload, download):
        self.per_round[server_round] = {
            "upload_gb": upload,
            "download_gb": download,
            "total_gb": upload + download
        }
        self.total_upload += upload
        self.total_download += download

    def get_total_cost(self):
        return {
            "total_upload_gb": self.total_upload,
            "total_download_gb": self.total_download,
            "total_communication_gb": self.total_upload + self.total_download
        }
