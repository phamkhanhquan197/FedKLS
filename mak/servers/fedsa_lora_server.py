from mak.servers.custom_server import ServerSaveData


class FedSALoRAServer(ServerSaveData):
    """FedSA-LoRA Server skeleton (logging and orchestration in ServerSaveData)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)