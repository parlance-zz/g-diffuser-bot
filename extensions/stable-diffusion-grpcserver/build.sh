#!/bin/bash

git submodule update --init --recursive
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/engines.proto
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/dashboard.proto
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated api-interfaces/src/proto/completion.proto

# Our generation.proto is adjusted.
python -m grpc_tools.protoc -Isdgrpcserver/proto -Iapi-interfaces/src/proto --python_out=sdgrpcserver/generated --grpc_python_out=sdgrpcserver/generated sdgrpcserver/proto/generation.proto
