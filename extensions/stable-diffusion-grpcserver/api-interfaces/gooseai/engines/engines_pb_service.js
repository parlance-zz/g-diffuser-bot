// package: gooseai
// file: engines.proto

var engines_pb = require("./engines_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var EnginesService = (function () {
  function EnginesService() {}
  EnginesService.serviceName = "gooseai.EnginesService";
  return EnginesService;
}());

EnginesService.ListEngines = {
  methodName: "ListEngines",
  service: EnginesService,
  requestStream: false,
  responseStream: false,
  requestType: engines_pb.ListEnginesRequest,
  responseType: engines_pb.Engines
};

exports.EnginesService = EnginesService;

function EnginesServiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

EnginesServiceClient.prototype.listEngines = function listEngines(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(EnginesService.ListEngines, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

exports.EnginesServiceClient = EnginesServiceClient;

