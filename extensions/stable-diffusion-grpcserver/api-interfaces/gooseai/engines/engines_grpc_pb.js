// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('grpc');
var engines_pb = require('./engines_pb.js');

function serialize_gooseai_Engines(arg) {
  if (!(arg instanceof engines_pb.Engines)) {
    throw new Error('Expected argument of type gooseai.Engines');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Engines(buffer_arg) {
  return engines_pb.Engines.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_ListEnginesRequest(arg) {
  if (!(arg instanceof engines_pb.ListEnginesRequest)) {
    throw new Error('Expected argument of type gooseai.ListEnginesRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_ListEnginesRequest(buffer_arg) {
  return engines_pb.ListEnginesRequest.deserializeBinary(new Uint8Array(buffer_arg));
}


var EnginesServiceService = exports.EnginesServiceService = {
  listEngines: {
    path: '/gooseai.EnginesService/ListEngines',
    requestStream: false,
    responseStream: false,
    requestType: engines_pb.ListEnginesRequest,
    responseType: engines_pb.Engines,
    requestSerialize: serialize_gooseai_ListEnginesRequest,
    requestDeserialize: deserialize_gooseai_ListEnginesRequest,
    responseSerialize: serialize_gooseai_Engines,
    responseDeserialize: deserialize_gooseai_Engines,
  },
};

exports.EnginesServiceClient = grpc.makeGenericClientConstructor(EnginesServiceService);
