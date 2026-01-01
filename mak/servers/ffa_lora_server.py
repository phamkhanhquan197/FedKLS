from mak.servers.custom_server import ServerSaveData


class FFALoRAServer(ServerSaveData):
    """FFA-LoRA Server skeleton.

    This integrates with FedKLS logging via ServerSaveData.
    Most of the FFA-LoRA logic lives in FFALoRAStrategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
