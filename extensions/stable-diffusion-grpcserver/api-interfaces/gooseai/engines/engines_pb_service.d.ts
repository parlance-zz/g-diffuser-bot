// package: gooseai
// file: engines.proto

import * as engines_pb from "./engines_pb";
import {grpc} from "@improbable-eng/grpc-web";

type EnginesServiceListEngines = {
  readonly methodName: string;
  readonly service: typeof EnginesService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof engines_pb.ListEnginesRequest;
  readonly responseType: typeof engines_pb.Engines;
};

export class EnginesService {
  static readonly serviceName: string;
  static readonly ListEngines: EnginesServiceListEngines;
}

export type ServiceError = { message: string, code: number; metadata: grpc.Metadata }
export type Status = { details: string, code: number; metadata: grpc.Metadata }

interface UnaryResponse {
  cancel(): void;
}
interface ResponseStream<T> {
  cancel(): void;
  on(type: 'data', handler: (message: T) => void): ResponseStream<T>;
  on(type: 'end', handler: (status?: Status) => void): ResponseStream<T>;
  on(type: 'status', handler: (status: Status) => void): ResponseStream<T>;
}
interface RequestStream<T> {
  write(message: T): RequestStream<T>;
  end(): void;
  cancel(): void;
  on(type: 'end', handler: (status?: Status) => void): RequestStream<T>;
  on(type: 'status', handler: (status: Status) => void): RequestStream<T>;
}
interface BidirectionalStream<ReqT, ResT> {
  write(message: ReqT): BidirectionalStream<ReqT, ResT>;
  end(): void;
  cancel(): void;
  on(type: 'data', handler: (message: ResT) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'end', handler: (status?: Status) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'status', handler: (status: Status) => void): BidirectionalStream<ReqT, ResT>;
}

export class EnginesServiceClient {
  readonly serviceHost: string;

  constructor(serviceHost: string, options?: grpc.RpcOptions);
  listEngines(
    requestMessage: engines_pb.ListEnginesRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: engines_pb.Engines|null) => void
  ): UnaryResponse;
  listEngines(
    requestMessage: engines_pb.ListEnginesRequest,
    callback: (error: ServiceError|null, responseMessage: engines_pb.Engines|null) => void
  ): UnaryResponse;
}

