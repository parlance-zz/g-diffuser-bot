// GENERATED CODE -- DO NOT EDIT!

// package: gooseai
// file: engines.proto

import * as engines_pb from "./engines_pb";
import * as grpc from "grpc";

interface IEnginesServiceService extends grpc.ServiceDefinition<grpc.UntypedServiceImplementation> {
  listEngines: grpc.MethodDefinition<engines_pb.ListEnginesRequest, engines_pb.Engines>;
}

export const EnginesServiceService: IEnginesServiceService;

export interface IEnginesServiceServer extends grpc.UntypedServiceImplementation {
  listEngines: grpc.handleUnaryCall<engines_pb.ListEnginesRequest, engines_pb.Engines>;
}

export class EnginesServiceClient extends grpc.Client {
  constructor(address: string, credentials: grpc.ChannelCredentials, options?: object);
  listEngines(argument: engines_pb.ListEnginesRequest, callback: grpc.requestCallback<engines_pb.Engines>): grpc.ClientUnaryCall;
  listEngines(argument: engines_pb.ListEnginesRequest, metadataOrOptions: grpc.Metadata | grpc.CallOptions | null, callback: grpc.requestCallback<engines_pb.Engines>): grpc.ClientUnaryCall;
  listEngines(argument: engines_pb.ListEnginesRequest, metadata: grpc.Metadata | null, options: grpc.CallOptions | null, callback: grpc.requestCallback<engines_pb.Engines>): grpc.ClientUnaryCall;
}
