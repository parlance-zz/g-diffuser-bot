import importlib
import sys
from pathlib import Path

base_path = __file__.replace("\\__init__.py", "")
sys.path.append(base_path) # todo: temporary, sorry if you found this :(
grpc_client = importlib.import_module(".client", package="stable-diffusion-grpcserver")

from .g_diffuser_utilities import *
