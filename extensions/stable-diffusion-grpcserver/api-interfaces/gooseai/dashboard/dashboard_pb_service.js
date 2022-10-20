// package: gooseai
// file: dashboard.proto

var dashboard_pb = require("./dashboard_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var DashboardService = (function () {
  function DashboardService() {}
  DashboardService.serviceName = "gooseai.DashboardService";
  return DashboardService;
}());

DashboardService.GetMe = {
  methodName: "GetMe",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.EmptyRequest,
  responseType: dashboard_pb.User
};

DashboardService.GetOrganization = {
  methodName: "GetOrganization",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.GetOrganizationRequest,
  responseType: dashboard_pb.Organization
};

DashboardService.GetMetrics = {
  methodName: "GetMetrics",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.GetMetricsRequest,
  responseType: dashboard_pb.Metrics
};

DashboardService.CreateAPIKey = {
  methodName: "CreateAPIKey",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.APIKeyRequest,
  responseType: dashboard_pb.APIKey
};

DashboardService.DeleteAPIKey = {
  methodName: "DeleteAPIKey",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.APIKeyFindRequest,
  responseType: dashboard_pb.APIKey
};

DashboardService.UpdateDefaultOrganization = {
  methodName: "UpdateDefaultOrganization",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.UpdateDefaultOrganizationRequest,
  responseType: dashboard_pb.User
};

DashboardService.GetClientSettings = {
  methodName: "GetClientSettings",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.EmptyRequest,
  responseType: dashboard_pb.ClientSettings
};

DashboardService.SetClientSettings = {
  methodName: "SetClientSettings",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.ClientSettings,
  responseType: dashboard_pb.ClientSettings
};

DashboardService.UpdateUserInfo = {
  methodName: "UpdateUserInfo",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.UpdateUserInfoRequest,
  responseType: dashboard_pb.User
};

DashboardService.CreatePasswordChangeTicket = {
  methodName: "CreatePasswordChangeTicket",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.EmptyRequest,
  responseType: dashboard_pb.UserPasswordChangeTicket
};

DashboardService.CreateCharge = {
  methodName: "CreateCharge",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.CreateChargeRequest,
  responseType: dashboard_pb.Charge
};

DashboardService.GetCharges = {
  methodName: "GetCharges",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.GetChargesRequest,
  responseType: dashboard_pb.Charges
};

DashboardService.CreateAutoChargeIntent = {
  methodName: "CreateAutoChargeIntent",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.CreateAutoChargeIntentRequest,
  responseType: dashboard_pb.AutoChargeIntent
};

DashboardService.UpdateAutoChargeIntent = {
  methodName: "UpdateAutoChargeIntent",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.CreateAutoChargeIntentRequest,
  responseType: dashboard_pb.AutoChargeIntent
};

DashboardService.GetAutoChargeIntent = {
  methodName: "GetAutoChargeIntent",
  service: DashboardService,
  requestStream: false,
  responseStream: false,
  requestType: dashboard_pb.GetAutoChargeRequest,
  responseType: dashboard_pb.AutoChargeIntent
};

exports.DashboardService = DashboardService;

function DashboardServiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

DashboardServiceClient.prototype.getMe = function getMe(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetMe, {
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

DashboardServiceClient.prototype.getOrganization = function getOrganization(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetOrganization, {
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

DashboardServiceClient.prototype.getMetrics = function getMetrics(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetMetrics, {
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

DashboardServiceClient.prototype.createAPIKey = function createAPIKey(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.CreateAPIKey, {
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

DashboardServiceClient.prototype.deleteAPIKey = function deleteAPIKey(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.DeleteAPIKey, {
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

DashboardServiceClient.prototype.updateDefaultOrganization = function updateDefaultOrganization(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.UpdateDefaultOrganization, {
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

DashboardServiceClient.prototype.getClientSettings = function getClientSettings(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetClientSettings, {
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

DashboardServiceClient.prototype.setClientSettings = function setClientSettings(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.SetClientSettings, {
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

DashboardServiceClient.prototype.updateUserInfo = function updateUserInfo(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.UpdateUserInfo, {
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

DashboardServiceClient.prototype.createPasswordChangeTicket = function createPasswordChangeTicket(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.CreatePasswordChangeTicket, {
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

DashboardServiceClient.prototype.createCharge = function createCharge(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.CreateCharge, {
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

DashboardServiceClient.prototype.getCharges = function getCharges(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetCharges, {
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

DashboardServiceClient.prototype.createAutoChargeIntent = function createAutoChargeIntent(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.CreateAutoChargeIntent, {
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

DashboardServiceClient.prototype.updateAutoChargeIntent = function updateAutoChargeIntent(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.UpdateAutoChargeIntent, {
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

DashboardServiceClient.prototype.getAutoChargeIntent = function getAutoChargeIntent(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(DashboardService.GetAutoChargeIntent, {
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

exports.DashboardServiceClient = DashboardServiceClient;

