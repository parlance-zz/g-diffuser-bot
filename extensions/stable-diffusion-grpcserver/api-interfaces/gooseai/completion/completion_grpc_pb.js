// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('grpc');
var completion_pb = require('./completion_pb.js');

function serialize_gooseai_Answer(arg) {
  if (!(arg instanceof completion_pb.Answer)) {
    throw new Error('Expected argument of type gooseai.Answer');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Answer(buffer_arg) {
  return completion_pb.Answer.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_Request(arg) {
  if (!(arg instanceof completion_pb.Request)) {
    throw new Error('Expected argument of type gooseai.Request');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Request(buffer_arg) {
  return completion_pb.Request.deserializeBinary(new Uint8Array(buffer_arg));
}


var CompletionServiceService = exports.CompletionServiceService = {
  completion: {
    path: '/gooseai.CompletionService/Completion',
    requestStream: false,
    responseStream: true,
    requestType: completion_pb.Request,
    responseType: completion_pb.Answer,
    requestSerialize: serialize_gooseai_Request,
    requestDeserialize: deserialize_gooseai_Request,
    responseSerialize: serialize_gooseai_Answer,
    responseDeserialize: deserialize_gooseai_Answer,
  },
};

exports.CompletionServiceClient = grpc.makeGenericClientConstructor(CompletionServiceService);
