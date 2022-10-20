// GENERATED CODE -- DO NOT EDIT!

// package: gooseai
// file: completion.proto

import * as completion_pb from "./completion_pb";
import * as grpc from "grpc";

interface ICompletionServiceService extends grpc.ServiceDefinition<grpc.UntypedServiceImplementation> {
  completion: grpc.MethodDefinition<completion_pb.Request, completion_pb.Answer>;
}

export const CompletionServiceService: ICompletionServiceService;

export interface ICompletionServiceServer extends grpc.UntypedServiceImplementation {
  completion: grpc.handleServerStreamingCall<completion_pb.Request, completion_pb.Answer>;
}

export class CompletionServiceClient extends grpc.Client {
  constructor(address: string, credentials: grpc.ChannelCredentials, options?: object);
  completion(argument: completion_pb.Request, metadataOrOptions?: grpc.Metadata | grpc.CallOptions | null): grpc.ClientReadableStream<completion_pb.Answer>;
  completion(argument: completion_pb.Request, metadata?: grpc.Metadata | null, options?: grpc.CallOptions | null): grpc.ClientReadableStream<completion_pb.Answer>;
}
