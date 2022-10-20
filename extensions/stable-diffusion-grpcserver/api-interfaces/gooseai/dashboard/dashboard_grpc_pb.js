// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('grpc');
var dashboard_pb = require('./dashboard_pb.js');

function serialize_gooseai_APIKey(arg) {
  if (!(arg instanceof dashboard_pb.APIKey)) {
    throw new Error('Expected argument of type gooseai.APIKey');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_APIKey(buffer_arg) {
  return dashboard_pb.APIKey.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_APIKeyFindRequest(arg) {
  if (!(arg instanceof dashboard_pb.APIKeyFindRequest)) {
    throw new Error('Expected argument of type gooseai.APIKeyFindRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_APIKeyFindRequest(buffer_arg) {
  return dashboard_pb.APIKeyFindRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_APIKeyRequest(arg) {
  if (!(arg instanceof dashboard_pb.APIKeyRequest)) {
    throw new Error('Expected argument of type gooseai.APIKeyRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_APIKeyRequest(buffer_arg) {
  return dashboard_pb.APIKeyRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_AutoChargeIntent(arg) {
  if (!(arg instanceof dashboard_pb.AutoChargeIntent)) {
    throw new Error('Expected argument of type gooseai.AutoChargeIntent');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_AutoChargeIntent(buffer_arg) {
  return dashboard_pb.AutoChargeIntent.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_Charge(arg) {
  if (!(arg instanceof dashboard_pb.Charge)) {
    throw new Error('Expected argument of type gooseai.Charge');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Charge(buffer_arg) {
  return dashboard_pb.Charge.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_Charges(arg) {
  if (!(arg instanceof dashboard_pb.Charges)) {
    throw new Error('Expected argument of type gooseai.Charges');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Charges(buffer_arg) {
  return dashboard_pb.Charges.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_ClientSettings(arg) {
  if (!(arg instanceof dashboard_pb.ClientSettings)) {
    throw new Error('Expected argument of type gooseai.ClientSettings');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_ClientSettings(buffer_arg) {
  return dashboard_pb.ClientSettings.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_CreateAutoChargeIntentRequest(arg) {
  if (!(arg instanceof dashboard_pb.CreateAutoChargeIntentRequest)) {
    throw new Error('Expected argument of type gooseai.CreateAutoChargeIntentRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_CreateAutoChargeIntentRequest(buffer_arg) {
  return dashboard_pb.CreateAutoChargeIntentRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_CreateChargeRequest(arg) {
  if (!(arg instanceof dashboard_pb.CreateChargeRequest)) {
    throw new Error('Expected argument of type gooseai.CreateChargeRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_CreateChargeRequest(buffer_arg) {
  return dashboard_pb.CreateChargeRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_EmptyRequest(arg) {
  if (!(arg instanceof dashboard_pb.EmptyRequest)) {
    throw new Error('Expected argument of type gooseai.EmptyRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_EmptyRequest(buffer_arg) {
  return dashboard_pb.EmptyRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_GetAutoChargeRequest(arg) {
  if (!(arg instanceof dashboard_pb.GetAutoChargeRequest)) {
    throw new Error('Expected argument of type gooseai.GetAutoChargeRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_GetAutoChargeRequest(buffer_arg) {
  return dashboard_pb.GetAutoChargeRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_GetChargesRequest(arg) {
  if (!(arg instanceof dashboard_pb.GetChargesRequest)) {
    throw new Error('Expected argument of type gooseai.GetChargesRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_GetChargesRequest(buffer_arg) {
  return dashboard_pb.GetChargesRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_GetMetricsRequest(arg) {
  if (!(arg instanceof dashboard_pb.GetMetricsRequest)) {
    throw new Error('Expected argument of type gooseai.GetMetricsRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_GetMetricsRequest(buffer_arg) {
  return dashboard_pb.GetMetricsRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_GetOrganizationRequest(arg) {
  if (!(arg instanceof dashboard_pb.GetOrganizationRequest)) {
    throw new Error('Expected argument of type gooseai.GetOrganizationRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_GetOrganizationRequest(buffer_arg) {
  return dashboard_pb.GetOrganizationRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_Metrics(arg) {
  if (!(arg instanceof dashboard_pb.Metrics)) {
    throw new Error('Expected argument of type gooseai.Metrics');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Metrics(buffer_arg) {
  return dashboard_pb.Metrics.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_Organization(arg) {
  if (!(arg instanceof dashboard_pb.Organization)) {
    throw new Error('Expected argument of type gooseai.Organization');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_Organization(buffer_arg) {
  return dashboard_pb.Organization.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_UpdateDefaultOrganizationRequest(arg) {
  if (!(arg instanceof dashboard_pb.UpdateDefaultOrganizationRequest)) {
    throw new Error('Expected argument of type gooseai.UpdateDefaultOrganizationRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_UpdateDefaultOrganizationRequest(buffer_arg) {
  return dashboard_pb.UpdateDefaultOrganizationRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_UpdateUserInfoRequest(arg) {
  if (!(arg instanceof dashboard_pb.UpdateUserInfoRequest)) {
    throw new Error('Expected argument of type gooseai.UpdateUserInfoRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_UpdateUserInfoRequest(buffer_arg) {
  return dashboard_pb.UpdateUserInfoRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_User(arg) {
  if (!(arg instanceof dashboard_pb.User)) {
    throw new Error('Expected argument of type gooseai.User');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_User(buffer_arg) {
  return dashboard_pb.User.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_gooseai_UserPasswordChangeTicket(arg) {
  if (!(arg instanceof dashboard_pb.UserPasswordChangeTicket)) {
    throw new Error('Expected argument of type gooseai.UserPasswordChangeTicket');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_gooseai_UserPasswordChangeTicket(buffer_arg) {
  return dashboard_pb.UserPasswordChangeTicket.deserializeBinary(new Uint8Array(buffer_arg));
}


var DashboardServiceService = exports.DashboardServiceService = {
  // Get info
getMe: {
    path: '/gooseai.DashboardService/GetMe',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.EmptyRequest,
    responseType: dashboard_pb.User,
    requestSerialize: serialize_gooseai_EmptyRequest,
    requestDeserialize: deserialize_gooseai_EmptyRequest,
    responseSerialize: serialize_gooseai_User,
    responseDeserialize: deserialize_gooseai_User,
  },
  getOrganization: {
    path: '/gooseai.DashboardService/GetOrganization',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.GetOrganizationRequest,
    responseType: dashboard_pb.Organization,
    requestSerialize: serialize_gooseai_GetOrganizationRequest,
    requestDeserialize: deserialize_gooseai_GetOrganizationRequest,
    responseSerialize: serialize_gooseai_Organization,
    responseDeserialize: deserialize_gooseai_Organization,
  },
  getMetrics: {
    path: '/gooseai.DashboardService/GetMetrics',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.GetMetricsRequest,
    responseType: dashboard_pb.Metrics,
    requestSerialize: serialize_gooseai_GetMetricsRequest,
    requestDeserialize: deserialize_gooseai_GetMetricsRequest,
    responseSerialize: serialize_gooseai_Metrics,
    responseDeserialize: deserialize_gooseai_Metrics,
  },
  // API key management
createAPIKey: {
    path: '/gooseai.DashboardService/CreateAPIKey',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.APIKeyRequest,
    responseType: dashboard_pb.APIKey,
    requestSerialize: serialize_gooseai_APIKeyRequest,
    requestDeserialize: deserialize_gooseai_APIKeyRequest,
    responseSerialize: serialize_gooseai_APIKey,
    responseDeserialize: deserialize_gooseai_APIKey,
  },
  deleteAPIKey: {
    path: '/gooseai.DashboardService/DeleteAPIKey',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.APIKeyFindRequest,
    responseType: dashboard_pb.APIKey,
    requestSerialize: serialize_gooseai_APIKeyFindRequest,
    requestDeserialize: deserialize_gooseai_APIKeyFindRequest,
    responseSerialize: serialize_gooseai_APIKey,
    responseDeserialize: deserialize_gooseai_APIKey,
  },
  // User settings
updateDefaultOrganization: {
    path: '/gooseai.DashboardService/UpdateDefaultOrganization',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.UpdateDefaultOrganizationRequest,
    responseType: dashboard_pb.User,
    requestSerialize: serialize_gooseai_UpdateDefaultOrganizationRequest,
    requestDeserialize: deserialize_gooseai_UpdateDefaultOrganizationRequest,
    responseSerialize: serialize_gooseai_User,
    responseDeserialize: deserialize_gooseai_User,
  },
  getClientSettings: {
    path: '/gooseai.DashboardService/GetClientSettings',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.EmptyRequest,
    responseType: dashboard_pb.ClientSettings,
    requestSerialize: serialize_gooseai_EmptyRequest,
    requestDeserialize: deserialize_gooseai_EmptyRequest,
    responseSerialize: serialize_gooseai_ClientSettings,
    responseDeserialize: deserialize_gooseai_ClientSettings,
  },
  setClientSettings: {
    path: '/gooseai.DashboardService/SetClientSettings',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.ClientSettings,
    responseType: dashboard_pb.ClientSettings,
    requestSerialize: serialize_gooseai_ClientSettings,
    requestDeserialize: deserialize_gooseai_ClientSettings,
    responseSerialize: serialize_gooseai_ClientSettings,
    responseDeserialize: deserialize_gooseai_ClientSettings,
  },
  updateUserInfo: {
    path: '/gooseai.DashboardService/UpdateUserInfo',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.UpdateUserInfoRequest,
    responseType: dashboard_pb.User,
    requestSerialize: serialize_gooseai_UpdateUserInfoRequest,
    requestDeserialize: deserialize_gooseai_UpdateUserInfoRequest,
    responseSerialize: serialize_gooseai_User,
    responseDeserialize: deserialize_gooseai_User,
  },
  createPasswordChangeTicket: {
    path: '/gooseai.DashboardService/CreatePasswordChangeTicket',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.EmptyRequest,
    responseType: dashboard_pb.UserPasswordChangeTicket,
    requestSerialize: serialize_gooseai_EmptyRequest,
    requestDeserialize: deserialize_gooseai_EmptyRequest,
    responseSerialize: serialize_gooseai_UserPasswordChangeTicket,
    responseDeserialize: deserialize_gooseai_UserPasswordChangeTicket,
  },
  // Payment functions
createCharge: {
    path: '/gooseai.DashboardService/CreateCharge',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.CreateChargeRequest,
    responseType: dashboard_pb.Charge,
    requestSerialize: serialize_gooseai_CreateChargeRequest,
    requestDeserialize: deserialize_gooseai_CreateChargeRequest,
    responseSerialize: serialize_gooseai_Charge,
    responseDeserialize: deserialize_gooseai_Charge,
  },
  getCharges: {
    path: '/gooseai.DashboardService/GetCharges',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.GetChargesRequest,
    responseType: dashboard_pb.Charges,
    requestSerialize: serialize_gooseai_GetChargesRequest,
    requestDeserialize: deserialize_gooseai_GetChargesRequest,
    responseSerialize: serialize_gooseai_Charges,
    responseDeserialize: deserialize_gooseai_Charges,
  },
  createAutoChargeIntent: {
    path: '/gooseai.DashboardService/CreateAutoChargeIntent',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.CreateAutoChargeIntentRequest,
    responseType: dashboard_pb.AutoChargeIntent,
    requestSerialize: serialize_gooseai_CreateAutoChargeIntentRequest,
    requestDeserialize: deserialize_gooseai_CreateAutoChargeIntentRequest,
    responseSerialize: serialize_gooseai_AutoChargeIntent,
    responseDeserialize: deserialize_gooseai_AutoChargeIntent,
  },
  updateAutoChargeIntent: {
    path: '/gooseai.DashboardService/UpdateAutoChargeIntent',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.CreateAutoChargeIntentRequest,
    responseType: dashboard_pb.AutoChargeIntent,
    requestSerialize: serialize_gooseai_CreateAutoChargeIntentRequest,
    requestDeserialize: deserialize_gooseai_CreateAutoChargeIntentRequest,
    responseSerialize: serialize_gooseai_AutoChargeIntent,
    responseDeserialize: deserialize_gooseai_AutoChargeIntent,
  },
  getAutoChargeIntent: {
    path: '/gooseai.DashboardService/GetAutoChargeIntent',
    requestStream: false,
    responseStream: false,
    requestType: dashboard_pb.GetAutoChargeRequest,
    responseType: dashboard_pb.AutoChargeIntent,
    requestSerialize: serialize_gooseai_GetAutoChargeRequest,
    requestDeserialize: deserialize_gooseai_GetAutoChargeRequest,
    responseSerialize: serialize_gooseai_AutoChargeIntent,
    responseDeserialize: deserialize_gooseai_AutoChargeIntent,
  },
};

exports.DashboardServiceClient = grpc.makeGenericClientConstructor(DashboardServiceService);
