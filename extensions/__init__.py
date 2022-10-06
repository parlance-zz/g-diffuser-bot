import importlib
import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(base_path)
sys.path.append(base_path+"/stable-diffusion-grpcserver")

#grpc_server = importlib.import_module(".server", package="sdgrpcserver")
grpc_client = importlib.import_module(".client", package="stable-diffusion-grpcserver")

from .g_diffuser_utilities import *
