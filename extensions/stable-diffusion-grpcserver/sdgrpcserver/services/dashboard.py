import dashboard_pb2, dashboard_pb2_grpc

class DashboardServiceServicer(dashboard_pb2_grpc.DashboardServiceServicer):
    def __init__(self):
        pass

    def GetMe(self, request, context):
        user = dashboard_pb2.User()
        user.id="0000-0000-0000-0001"
        return user
