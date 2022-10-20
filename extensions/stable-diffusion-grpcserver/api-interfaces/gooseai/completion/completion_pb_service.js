// package: gooseai
// file: completion.proto

var completion_pb = require("./completion_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var CompletionService = (function () {
  function CompletionService() {}
  CompletionService.serviceName = "gooseai.CompletionService";
  return CompletionService;
}());

CompletionService.Completion = {
  methodName: "Completion",
  service: CompletionService,
  requestStream: false,
  responseStream: true,
  requestType: completion_pb.Request,
  responseType: completion_pb.Answer
};

exports.CompletionService = CompletionService;

function CompletionServiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

CompletionServiceClient.prototype.completion = function completion(requestMessage, metadata) {
  var listeners = {
    data: [],
    end: [],
    status: []
  };
  var client = grpc.invoke(CompletionService.Completion, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onMessage: function (responseMessage) {
      listeners.data.forEach(function (handler) {
        handler(responseMessage);
      });
    },
    onEnd: function (status, statusMessage, trailers) {
      listeners.status.forEach(function (handler) {
        handler({ code: status, details: statusMessage, metadata: trailers });
      });
      listeners.end.forEach(function (handler) {
        handler({ code: status, details: statusMessage, metadata: trailers });
      });
      listeners = null;
    }
  });
  return {
    on: function (type, handler) {
      listeners[type].push(handler);
      return this;
    },
    cancel: function () {
      listeners = null;
      client.close();
    }
  };
};

exports.CompletionServiceClient = CompletionServiceClient;

