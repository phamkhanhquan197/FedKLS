from mak.servers.custom_server import ServerSaveData


class FlexLoRAServer(ServerSaveData):
    """FlexLoRA server wrapper.

    Strict inheritance: only subclass ServerSaveData.
    No additional behavior required for the initial baseline;
    server-side algorithmic logic lives in the strategy.
    """

    pass

