from mak.servers.custom_server import ServerSaveData


class FlexLoRAServer(ServerSaveData):
    """FlexLoRA server wrapper (no extra behavior)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
