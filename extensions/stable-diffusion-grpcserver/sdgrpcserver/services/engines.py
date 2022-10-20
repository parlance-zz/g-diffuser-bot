import engines_pb2, engines_pb2_grpc

class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    def __init__(self, manager):
        self._manager = manager

    def ListEngines(self, request, context):
        engines = engines_pb2.Engines()

        status = self._manager.getStatus()
        for engine in self._manager.engines:
            if not (engine.get("enabled", False) or engine.get("visible", False)):
                continue

            info=engines_pb2.EngineInfo()
            info.id=engine["id"]
            info.name=engine["name"]
            info.description=engine["description"]
            info.owner="stable-diffusion-grpcserver"
            info.ready=status.get(engine["id"], False)
            info.type=engines_pb2.EngineType.PICTURE

            engines.engine.append(engine)

        return engines

