// package: gooseai
// file: dashboard.proto

import * as dashboard_pb from "./dashboard_pb";
import {grpc} from "@improbable-eng/grpc-web";

type DashboardServiceGetMe = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.EmptyRequest;
  readonly responseType: typeof dashboard_pb.User;
};

type DashboardServiceGetOrganization = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.GetOrganizationRequest;
  readonly responseType: typeof dashboard_pb.Organization;
};

type DashboardServiceGetMetrics = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.GetMetricsRequest;
  readonly responseType: typeof dashboard_pb.Metrics;
};

type DashboardServiceCreateAPIKey = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.APIKeyRequest;
  readonly responseType: typeof dashboard_pb.APIKey;
};

type DashboardServiceDeleteAPIKey = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.APIKeyFindRequest;
  readonly responseType: typeof dashboard_pb.APIKey;
};

type DashboardServiceUpdateDefaultOrganization = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.UpdateDefaultOrganizationRequest;
  readonly responseType: typeof dashboard_pb.User;
};

type DashboardServiceGetClientSettings = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.EmptyRequest;
  readonly responseType: typeof dashboard_pb.ClientSettings;
};

type DashboardServiceSetClientSettings = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.ClientSettings;
  readonly responseType: typeof dashboard_pb.ClientSettings;
};

type DashboardServiceUpdateUserInfo = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.UpdateUserInfoRequest;
  readonly responseType: typeof dashboard_pb.User;
};

type DashboardServiceCreatePasswordChangeTicket = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.EmptyRequest;
  readonly responseType: typeof dashboard_pb.UserPasswordChangeTicket;
};

type DashboardServiceCreateCharge = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.CreateChargeRequest;
  readonly responseType: typeof dashboard_pb.Charge;
};

type DashboardServiceGetCharges = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.GetChargesRequest;
  readonly responseType: typeof dashboard_pb.Charges;
};

type DashboardServiceCreateAutoChargeIntent = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.CreateAutoChargeIntentRequest;
  readonly responseType: typeof dashboard_pb.AutoChargeIntent;
};

type DashboardServiceUpdateAutoChargeIntent = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.CreateAutoChargeIntentRequest;
  readonly responseType: typeof dashboard_pb.AutoChargeIntent;
};

type DashboardServiceGetAutoChargeIntent = {
  readonly methodName: string;
  readonly service: typeof DashboardService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof dashboard_pb.GetAutoChargeRequest;
  readonly responseType: typeof dashboard_pb.AutoChargeIntent;
};

export class DashboardService {
  static readonly serviceName: string;
  static readonly GetMe: DashboardServiceGetMe;
  static readonly GetOrganization: DashboardServiceGetOrganization;
  static readonly GetMetrics: DashboardServiceGetMetrics;
  static readonly CreateAPIKey: DashboardServiceCreateAPIKey;
  static readonly DeleteAPIKey: DashboardServiceDeleteAPIKey;
  static readonly UpdateDefaultOrganization: DashboardServiceUpdateDefaultOrganization;
  static readonly GetClientSettings: DashboardServiceGetClientSettings;
  static readonly SetClientSettings: DashboardServiceSetClientSettings;
  static readonly UpdateUserInfo: DashboardServiceUpdateUserInfo;
  static readonly CreatePasswordChangeTicket: DashboardServiceCreatePasswordChangeTicket;
  static readonly CreateCharge: DashboardServiceCreateCharge;
  static readonly GetCharges: DashboardServiceGetCharges;
  static readonly CreateAutoChargeIntent: DashboardServiceCreateAutoChargeIntent;
  static readonly UpdateAutoChargeIntent: DashboardServiceUpdateAutoChargeIntent;
  static readonly GetAutoChargeIntent: DashboardServiceGetAutoChargeIntent;
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

export class DashboardServiceClient {
  readonly serviceHost: string;

  constructor(serviceHost: string, options?: grpc.RpcOptions);
  getMe(
    requestMessage: dashboard_pb.EmptyRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  getMe(
    requestMessage: dashboard_pb.EmptyRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  getOrganization(
    requestMessage: dashboard_pb.GetOrganizationRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Organization|null) => void
  ): UnaryResponse;
  getOrganization(
    requestMessage: dashboard_pb.GetOrganizationRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Organization|null) => void
  ): UnaryResponse;
  getMetrics(
    requestMessage: dashboard_pb.GetMetricsRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Metrics|null) => void
  ): UnaryResponse;
  getMetrics(
    requestMessage: dashboard_pb.GetMetricsRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Metrics|null) => void
  ): UnaryResponse;
  createAPIKey(
    requestMessage: dashboard_pb.APIKeyRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.APIKey|null) => void
  ): UnaryResponse;
  createAPIKey(
    requestMessage: dashboard_pb.APIKeyRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.APIKey|null) => void
  ): UnaryResponse;
  deleteAPIKey(
    requestMessage: dashboard_pb.APIKeyFindRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.APIKey|null) => void
  ): UnaryResponse;
  deleteAPIKey(
    requestMessage: dashboard_pb.APIKeyFindRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.APIKey|null) => void
  ): UnaryResponse;
  updateDefaultOrganization(
    requestMessage: dashboard_pb.UpdateDefaultOrganizationRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  updateDefaultOrganization(
    requestMessage: dashboard_pb.UpdateDefaultOrganizationRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  getClientSettings(
    requestMessage: dashboard_pb.EmptyRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.ClientSettings|null) => void
  ): UnaryResponse;
  getClientSettings(
    requestMessage: dashboard_pb.EmptyRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.ClientSettings|null) => void
  ): UnaryResponse;
  setClientSettings(
    requestMessage: dashboard_pb.ClientSettings,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.ClientSettings|null) => void
  ): UnaryResponse;
  setClientSettings(
    requestMessage: dashboard_pb.ClientSettings,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.ClientSettings|null) => void
  ): UnaryResponse;
  updateUserInfo(
    requestMessage: dashboard_pb.UpdateUserInfoRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  updateUserInfo(
    requestMessage: dashboard_pb.UpdateUserInfoRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.User|null) => void
  ): UnaryResponse;
  createPasswordChangeTicket(
    requestMessage: dashboard_pb.EmptyRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.UserPasswordChangeTicket|null) => void
  ): UnaryResponse;
  createPasswordChangeTicket(
    requestMessage: dashboard_pb.EmptyRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.UserPasswordChangeTicket|null) => void
  ): UnaryResponse;
  createCharge(
    requestMessage: dashboard_pb.CreateChargeRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Charge|null) => void
  ): UnaryResponse;
  createCharge(
    requestMessage: dashboard_pb.CreateChargeRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Charge|null) => void
  ): UnaryResponse;
  getCharges(
    requestMessage: dashboard_pb.GetChargesRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Charges|null) => void
  ): UnaryResponse;
  getCharges(
    requestMessage: dashboard_pb.GetChargesRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.Charges|null) => void
  ): UnaryResponse;
  createAutoChargeIntent(
    requestMessage: dashboard_pb.CreateAutoChargeIntentRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
  createAutoChargeIntent(
    requestMessage: dashboard_pb.CreateAutoChargeIntentRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
  updateAutoChargeIntent(
    requestMessage: dashboard_pb.CreateAutoChargeIntentRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
  updateAutoChargeIntent(
    requestMessage: dashboard_pb.CreateAutoChargeIntentRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
  getAutoChargeIntent(
    requestMessage: dashboard_pb.GetAutoChargeRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
  getAutoChargeIntent(
    requestMessage: dashboard_pb.GetAutoChargeRequest,
    callback: (error: ServiceError|null, responseMessage: dashboard_pb.AutoChargeIntent|null) => void
  ): UnaryResponse;
}

