// package: gooseai
// file: completion.proto

import * as completion_pb from "./completion_pb";
import {grpc} from "@improbable-eng/grpc-web";

type CompletionServiceCompletion = {
  readonly methodName: string;
  readonly service: typeof CompletionService;
  readonly requestStream: false;
  readonly responseStream: true;
  readonly requestType: typeof completion_pb.Request;
  readonly responseType: typeof completion_pb.Answer;
};

export class CompletionService {
  static readonly serviceName: string;
  static readonly Completion: CompletionServiceCompletion;
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

export class CompletionServiceClient {
  readonly serviceHost: string;

  constructor(serviceHost: string, options?: grpc.RpcOptions);
  completion(requestMessage: completion_pb.Request, metadata?: grpc.Metadata): ResponseStream<completion_pb.Answer>;
}

