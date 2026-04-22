from mak.servers.custom_server import ServerSaveData

class FedAWAServer(ServerSaveData):
    """FedAWA server implementation inheriting from ServerSaveData."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)