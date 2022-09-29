import argparse, os, sys
from concurrent import futures

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import grpc
import hupper
from waitress import serve
from sdgrpcserver.sonora.wsgi import grpcWSGI
from wsgicors import CORS

# Google protoc compiler is dumb about imports (https://github.com/protocolbuffers/protobuf/issues/1491)
# TODO: Move to https://github.com/danielgtaylor/python-betterproto
generatedPath = os.path.join(os.path.dirname(__file__), "generated")
sys.path.append(generatedPath)

import generation_pb2_grpc, dashboard_pb2_grpc, engines_pb2_grpc

from sdgrpcserver.manager import EngineManager
from sdgrpcserver.services.dashboard import DashboardServiceServicer
from sdgrpcserver.services.generate import GenerationServiceServicer
from sdgrpcserver.services.engines import EnginesServiceServicer

class DartGRPCCompatibility(object):
    """Fixes a couple of compatibility issues between Dart GRPC-WEB and Sonora

    - Dart GRPC-WEB doesn't set HTTP_ACCEPT header, but Sonora needs it to build Content-Type header on response
    - Sonora sets Access-Control-Allow-Origin to HTTP_HOST, and we need to strip it out so CORSWSGI can set the correct value
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def wrapped_start_response(status, headers):
            headers = [header for header in headers if header[0] != 'Access-Control-Allow-Origin']
            return start_response(status, headers)
        
        if environ.get("HTTP_ACCEPT") == "*/*":
            environ["HTTP_ACCEPT"] = "application/grpc-web+proto"

        return self.app(environ, wrapped_start_response)

def start(manager, listen):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(manager), server)
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), server)
    engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(manager), server)

    server.add_insecure_port('[::]:50051')
    server.start()

    grpcapp = wsgi_app = grpcWSGI(None)
    wsgi_app = DartGRPCCompatibility(wsgi_app)
    wsgi_app = CORS(wsgi_app, headers="*", methods="*", origin="*")

    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(manager), grpcapp)
    dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), grpcapp)
    engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(manager), grpcapp)

    print("Ready, GRPC listening on port 50051, GRPC-Web listening on port 5000")
    serve(wsgi_app, listen=listen)

    #This does same thing as app.run
    #server.wait_for_termination()


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enginecfg", "-E", type=str, default="./engines.yaml", help="Path to the engines.yaml file"
    )
    parser.add_argument(
        "--listen_to_all", "-L", action='store_true', help="Accept requests from the local network, not just localhost" 
    )
    parser.add_argument(
        "--enable_mps", type=bool, default=False, help="Use MPS on MacOS where available"
    )
    parser.add_argument(
        "--vram_optimisation_level", "-V", type=int, default=2, help="How much to trade off performance to reduce VRAM usage (0 = none, 2 = max)"
    )
    parser.add_argument(
        "--nsfw_behaviour", "-N", type=str, default="block", choices=["block", "flag"], help="What to do with images detected as NSFW"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on source change"
    )
    parser.add_argument(
        "--models_root", "-W", type=str, default="./models", help="Root path for models referenced by engine.yaml config"
    )
    args = parser.parse_args()
    args = argparse.Namespace(**(vars(args) | kwargs)) # merge with keyword args for convenience
    
    #if args.reload:
        # start_reloader will only return in a monitored subprocess
        #reloader = hupper.start_reloader('sdgrpcserver.server.main', reload_interval=10)

    with open(os.path.normpath(args.enginecfg), 'r') as cfg:
        engines = yaml.load(cfg, Loader=Loader)
        manager = EngineManager(engines=engines, models_root=args.models_root, enable_mps=args.enable_mps, vram_optimisation_level=args.vram_optimisation_level, nsfw_behaviour=args.nsfw_behaviour)

        start(manager, "*:5000" if args.listen_to_all else "localhost:5000")

    sys.exit(-1)

