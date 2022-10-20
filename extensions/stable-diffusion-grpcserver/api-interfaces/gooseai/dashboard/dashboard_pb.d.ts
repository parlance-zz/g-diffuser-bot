// package: gooseai
// file: dashboard.proto

import * as jspb from "google-protobuf";

export class OrganizationMember extends jspb.Message {
  hasOrganization(): boolean;
  clearOrganization(): void;
  getOrganization(): Organization | undefined;
  setOrganization(value?: Organization): void;

  hasUser(): boolean;
  clearUser(): void;
  getUser(): User | undefined;
  setUser(value?: User): void;

  getRole(): OrganizationRoleMap[keyof OrganizationRoleMap];
  setRole(value: OrganizationRoleMap[keyof OrganizationRoleMap]): void;

  getIsDefault(): boolean;
  setIsDefault(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): OrganizationMember.AsObject;
  static toObject(includeInstance: boolean, msg: OrganizationMember): OrganizationMember.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: OrganizationMember, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): OrganizationMember;
  static deserializeBinaryFromReader(message: OrganizationMember, reader: jspb.BinaryReader): OrganizationMember;
}

export namespace OrganizationMember {
  export type AsObject = {
    organization?: Organization.AsObject,
    user?: User.AsObject,
    role: OrganizationRoleMap[keyof OrganizationRoleMap],
    isDefault: boolean,
  }
}

export class OrganizationGrant extends jspb.Message {
  getAmountGranted(): number;
  setAmountGranted(value: number): void;

  getAmountUsed(): number;
  setAmountUsed(value: number): void;

  getExpiresAt(): number;
  setExpiresAt(value: number): void;

  getGrantedAt(): number;
  setGrantedAt(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): OrganizationGrant.AsObject;
  static toObject(includeInstance: boolean, msg: OrganizationGrant): OrganizationGrant.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: OrganizationGrant, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): OrganizationGrant;
  static deserializeBinaryFromReader(message: OrganizationGrant, reader: jspb.BinaryReader): OrganizationGrant;
}

export namespace OrganizationGrant {
  export type AsObject = {
    amountGranted: number,
    amountUsed: number,
    expiresAt: number,
    grantedAt: number,
  }
}

export class OrganizationPaymentInfo extends jspb.Message {
  getBalance(): number;
  setBalance(value: number): void;

  clearGrantsList(): void;
  getGrantsList(): Array<OrganizationGrant>;
  setGrantsList(value: Array<OrganizationGrant>): void;
  addGrants(value?: OrganizationGrant, index?: number): OrganizationGrant;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): OrganizationPaymentInfo.AsObject;
  static toObject(includeInstance: boolean, msg: OrganizationPaymentInfo): OrganizationPaymentInfo.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: OrganizationPaymentInfo, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): OrganizationPaymentInfo;
  static deserializeBinaryFromReader(message: OrganizationPaymentInfo, reader: jspb.BinaryReader): OrganizationPaymentInfo;
}

export namespace OrganizationPaymentInfo {
  export type AsObject = {
    balance: number,
    grantsList: Array<OrganizationGrant.AsObject>,
  }
}

export class OrganizationAutoCharge extends jspb.Message {
  getEnabled(): boolean;
  setEnabled(value: boolean): void;

  getId(): string;
  setId(value: string): void;

  getCreatedAt(): number;
  setCreatedAt(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): OrganizationAutoCharge.AsObject;
  static toObject(includeInstance: boolean, msg: OrganizationAutoCharge): OrganizationAutoCharge.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: OrganizationAutoCharge, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): OrganizationAutoCharge;
  static deserializeBinaryFromReader(message: OrganizationAutoCharge, reader: jspb.BinaryReader): OrganizationAutoCharge;
}

export namespace OrganizationAutoCharge {
  export type AsObject = {
    enabled: boolean,
    id: string,
    createdAt: number,
  }
}

export class Organization extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  getName(): string;
  setName(value: string): void;

  getDescription(): string;
  setDescription(value: string): void;

  clearMembersList(): void;
  getMembersList(): Array<OrganizationMember>;
  setMembersList(value: Array<OrganizationMember>): void;
  addMembers(value?: OrganizationMember, index?: number): OrganizationMember;

  hasPaymentInfo(): boolean;
  clearPaymentInfo(): void;
  getPaymentInfo(): OrganizationPaymentInfo | undefined;
  setPaymentInfo(value?: OrganizationPaymentInfo): void;

  hasStripeCustomerId(): boolean;
  clearStripeCustomerId(): void;
  getStripeCustomerId(): string;
  setStripeCustomerId(value: string): void;

  hasAutoCharge(): boolean;
  clearAutoCharge(): void;
  getAutoCharge(): OrganizationAutoCharge | undefined;
  setAutoCharge(value?: OrganizationAutoCharge): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Organization.AsObject;
  static toObject(includeInstance: boolean, msg: Organization): Organization.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Organization, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Organization;
  static deserializeBinaryFromReader(message: Organization, reader: jspb.BinaryReader): Organization;
}

export namespace Organization {
  export type AsObject = {
    id: string,
    name: string,
    description: string,
    membersList: Array<OrganizationMember.AsObject>,
    paymentInfo?: OrganizationPaymentInfo.AsObject,
    stripeCustomerId: string,
    autoCharge?: OrganizationAutoCharge.AsObject,
  }
}

export class APIKey extends jspb.Message {
  getKey(): string;
  setKey(value: string): void;

  getIsSecret(): boolean;
  setIsSecret(value: boolean): void;

  getCreatedAt(): number;
  setCreatedAt(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): APIKey.AsObject;
  static toObject(includeInstance: boolean, msg: APIKey): APIKey.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: APIKey, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): APIKey;
  static deserializeBinaryFromReader(message: APIKey, reader: jspb.BinaryReader): APIKey;
}

export namespace APIKey {
  export type AsObject = {
    key: string,
    isSecret: boolean,
    createdAt: number,
  }
}

export class User extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  hasAuthId(): boolean;
  clearAuthId(): void;
  getAuthId(): string;
  setAuthId(value: string): void;

  getProfilePicture(): string;
  setProfilePicture(value: string): void;

  getEmail(): string;
  setEmail(value: string): void;

  clearOrganizationsList(): void;
  getOrganizationsList(): Array<OrganizationMember>;
  setOrganizationsList(value: Array<OrganizationMember>): void;
  addOrganizations(value?: OrganizationMember, index?: number): OrganizationMember;

  clearApiKeysList(): void;
  getApiKeysList(): Array<APIKey>;
  setApiKeysList(value: Array<APIKey>): void;
  addApiKeys(value?: APIKey, index?: number): APIKey;

  getCreatedAt(): number;
  setCreatedAt(value: number): void;

  hasEmailVerified(): boolean;
  clearEmailVerified(): void;
  getEmailVerified(): boolean;
  setEmailVerified(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): User.AsObject;
  static toObject(includeInstance: boolean, msg: User): User.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: User, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): User;
  static deserializeBinaryFromReader(message: User, reader: jspb.BinaryReader): User;
}

export namespace User {
  export type AsObject = {
    id: string,
    authId: string,
    profilePicture: string,
    email: string,
    organizationsList: Array<OrganizationMember.AsObject>,
    apiKeysList: Array<APIKey.AsObject>,
    createdAt: number,
    emailVerified: boolean,
  }
}

export class CostData extends jspb.Message {
  getAmountTokens(): number;
  setAmountTokens(value: number): void;

  getAmountCredits(): number;
  setAmountCredits(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CostData.AsObject;
  static toObject(includeInstance: boolean, msg: CostData): CostData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CostData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CostData;
  static deserializeBinaryFromReader(message: CostData, reader: jspb.BinaryReader): CostData;
}

export namespace CostData {
  export type AsObject = {
    amountTokens: number,
    amountCredits: number,
  }
}

export class UsageMetric extends jspb.Message {
  getOperation(): string;
  setOperation(value: string): void;

  getEngine(): string;
  setEngine(value: string): void;

  hasInputCost(): boolean;
  clearInputCost(): void;
  getInputCost(): CostData | undefined;
  setInputCost(value?: CostData): void;

  hasOutputCost(): boolean;
  clearOutputCost(): void;
  getOutputCost(): CostData | undefined;
  setOutputCost(value?: CostData): void;

  hasUser(): boolean;
  clearUser(): void;
  getUser(): string;
  setUser(value: string): void;

  getAggregationTimestamp(): number;
  setAggregationTimestamp(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): UsageMetric.AsObject;
  static toObject(includeInstance: boolean, msg: UsageMetric): UsageMetric.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: UsageMetric, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): UsageMetric;
  static deserializeBinaryFromReader(message: UsageMetric, reader: jspb.BinaryReader): UsageMetric;
}

export namespace UsageMetric {
  export type AsObject = {
    operation: string,
    engine: string,
    inputCost?: CostData.AsObject,
    outputCost?: CostData.AsObject,
    user: string,
    aggregationTimestamp: number,
  }
}

export class CostTotal extends jspb.Message {
  getAmountTokens(): number;
  setAmountTokens(value: number): void;

  getAmountCredits(): number;
  setAmountCredits(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CostTotal.AsObject;
  static toObject(includeInstance: boolean, msg: CostTotal): CostTotal.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CostTotal, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CostTotal;
  static deserializeBinaryFromReader(message: CostTotal, reader: jspb.BinaryReader): CostTotal;
}

export namespace CostTotal {
  export type AsObject = {
    amountTokens: number,
    amountCredits: number,
  }
}

export class TotalMetricsData extends jspb.Message {
  hasInputTotal(): boolean;
  clearInputTotal(): void;
  getInputTotal(): CostTotal | undefined;
  setInputTotal(value?: CostTotal): void;

  hasOutputTotal(): boolean;
  clearOutputTotal(): void;
  getOutputTotal(): CostTotal | undefined;
  setOutputTotal(value?: CostTotal): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TotalMetricsData.AsObject;
  static toObject(includeInstance: boolean, msg: TotalMetricsData): TotalMetricsData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: TotalMetricsData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): TotalMetricsData;
  static deserializeBinaryFromReader(message: TotalMetricsData, reader: jspb.BinaryReader): TotalMetricsData;
}

export namespace TotalMetricsData {
  export type AsObject = {
    inputTotal?: CostTotal.AsObject,
    outputTotal?: CostTotal.AsObject,
  }
}

export class Metrics extends jspb.Message {
  clearMetricsList(): void;
  getMetricsList(): Array<UsageMetric>;
  setMetricsList(value: Array<UsageMetric>): void;
  addMetrics(value?: UsageMetric, index?: number): UsageMetric;

  hasTotal(): boolean;
  clearTotal(): void;
  getTotal(): TotalMetricsData | undefined;
  setTotal(value?: TotalMetricsData): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Metrics.AsObject;
  static toObject(includeInstance: boolean, msg: Metrics): Metrics.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Metrics, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Metrics;
  static deserializeBinaryFromReader(message: Metrics, reader: jspb.BinaryReader): Metrics;
}

export namespace Metrics {
  export type AsObject = {
    metricsList: Array<UsageMetric.AsObject>,
    total?: TotalMetricsData.AsObject,
  }
}

export class EmptyRequest extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EmptyRequest.AsObject;
  static toObject(includeInstance: boolean, msg: EmptyRequest): EmptyRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EmptyRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EmptyRequest;
  static deserializeBinaryFromReader(message: EmptyRequest, reader: jspb.BinaryReader): EmptyRequest;
}

export namespace EmptyRequest {
  export type AsObject = {
  }
}

export class GetOrganizationRequest extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GetOrganizationRequest.AsObject;
  static toObject(includeInstance: boolean, msg: GetOrganizationRequest): GetOrganizationRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: GetOrganizationRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): GetOrganizationRequest;
  static deserializeBinaryFromReader(message: GetOrganizationRequest, reader: jspb.BinaryReader): GetOrganizationRequest;
}

export namespace GetOrganizationRequest {
  export type AsObject = {
    id: string,
  }
}

export class GetMetricsRequest extends jspb.Message {
  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  hasUserId(): boolean;
  clearUserId(): void;
  getUserId(): string;
  setUserId(value: string): void;

  getRangeFrom(): number;
  setRangeFrom(value: number): void;

  getRangeTo(): number;
  setRangeTo(value: number): void;

  getIncludePerRequestMetrics(): boolean;
  setIncludePerRequestMetrics(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GetMetricsRequest.AsObject;
  static toObject(includeInstance: boolean, msg: GetMetricsRequest): GetMetricsRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: GetMetricsRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): GetMetricsRequest;
  static deserializeBinaryFromReader(message: GetMetricsRequest, reader: jspb.BinaryReader): GetMetricsRequest;
}

export namespace GetMetricsRequest {
  export type AsObject = {
    organizationId: string,
    userId: string,
    rangeFrom: number,
    rangeTo: number,
    includePerRequestMetrics: boolean,
  }
}

export class APIKeyRequest extends jspb.Message {
  getIsSecret(): boolean;
  setIsSecret(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): APIKeyRequest.AsObject;
  static toObject(includeInstance: boolean, msg: APIKeyRequest): APIKeyRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: APIKeyRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): APIKeyRequest;
  static deserializeBinaryFromReader(message: APIKeyRequest, reader: jspb.BinaryReader): APIKeyRequest;
}

export namespace APIKeyRequest {
  export type AsObject = {
    isSecret: boolean,
  }
}

export class APIKeyFindRequest extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): APIKeyFindRequest.AsObject;
  static toObject(includeInstance: boolean, msg: APIKeyFindRequest): APIKeyFindRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: APIKeyFindRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): APIKeyFindRequest;
  static deserializeBinaryFromReader(message: APIKeyFindRequest, reader: jspb.BinaryReader): APIKeyFindRequest;
}

export namespace APIKeyFindRequest {
  export type AsObject = {
    id: string,
  }
}

export class UpdateDefaultOrganizationRequest extends jspb.Message {
  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): UpdateDefaultOrganizationRequest.AsObject;
  static toObject(includeInstance: boolean, msg: UpdateDefaultOrganizationRequest): UpdateDefaultOrganizationRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: UpdateDefaultOrganizationRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): UpdateDefaultOrganizationRequest;
  static deserializeBinaryFromReader(message: UpdateDefaultOrganizationRequest, reader: jspb.BinaryReader): UpdateDefaultOrganizationRequest;
}

export namespace UpdateDefaultOrganizationRequest {
  export type AsObject = {
    organizationId: string,
  }
}

export class ClientSettings extends jspb.Message {
  getSettings(): Uint8Array | string;
  getSettings_asU8(): Uint8Array;
  getSettings_asB64(): string;
  setSettings(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClientSettings.AsObject;
  static toObject(includeInstance: boolean, msg: ClientSettings): ClientSettings.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClientSettings, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClientSettings;
  static deserializeBinaryFromReader(message: ClientSettings, reader: jspb.BinaryReader): ClientSettings;
}

export namespace ClientSettings {
  export type AsObject = {
    settings: Uint8Array | string,
  }
}

export class CreateAutoChargeIntentRequest extends jspb.Message {
  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  getMonthlyMaximum(): number;
  setMonthlyMaximum(value: number): void;

  getMinimumValue(): number;
  setMinimumValue(value: number): void;

  getAmountCredits(): number;
  setAmountCredits(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CreateAutoChargeIntentRequest.AsObject;
  static toObject(includeInstance: boolean, msg: CreateAutoChargeIntentRequest): CreateAutoChargeIntentRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CreateAutoChargeIntentRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CreateAutoChargeIntentRequest;
  static deserializeBinaryFromReader(message: CreateAutoChargeIntentRequest, reader: jspb.BinaryReader): CreateAutoChargeIntentRequest;
}

export namespace CreateAutoChargeIntentRequest {
  export type AsObject = {
    organizationId: string,
    monthlyMaximum: number,
    minimumValue: number,
    amountCredits: number,
  }
}

export class CreateChargeRequest extends jspb.Message {
  getAmount(): number;
  setAmount(value: number): void;

  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CreateChargeRequest.AsObject;
  static toObject(includeInstance: boolean, msg: CreateChargeRequest): CreateChargeRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CreateChargeRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CreateChargeRequest;
  static deserializeBinaryFromReader(message: CreateChargeRequest, reader: jspb.BinaryReader): CreateChargeRequest;
}

export namespace CreateChargeRequest {
  export type AsObject = {
    amount: number,
    organizationId: string,
  }
}

export class GetChargesRequest extends jspb.Message {
  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  getRangeFrom(): number;
  setRangeFrom(value: number): void;

  getRangeTo(): number;
  setRangeTo(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GetChargesRequest.AsObject;
  static toObject(includeInstance: boolean, msg: GetChargesRequest): GetChargesRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: GetChargesRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): GetChargesRequest;
  static deserializeBinaryFromReader(message: GetChargesRequest, reader: jspb.BinaryReader): GetChargesRequest;
}

export namespace GetChargesRequest {
  export type AsObject = {
    organizationId: string,
    rangeFrom: number,
    rangeTo: number,
  }
}

export class Charge extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  getPaid(): boolean;
  setPaid(value: boolean): void;

  getReceiptLink(): string;
  setReceiptLink(value: string): void;

  getPaymentLink(): string;
  setPaymentLink(value: string): void;

  getCreatedAt(): number;
  setCreatedAt(value: number): void;

  getAmountCredits(): number;
  setAmountCredits(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Charge.AsObject;
  static toObject(includeInstance: boolean, msg: Charge): Charge.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Charge, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Charge;
  static deserializeBinaryFromReader(message: Charge, reader: jspb.BinaryReader): Charge;
}

export namespace Charge {
  export type AsObject = {
    id: string,
    paid: boolean,
    receiptLink: string,
    paymentLink: string,
    createdAt: number,
    amountCredits: number,
  }
}

export class Charges extends jspb.Message {
  clearChargesList(): void;
  getChargesList(): Array<Charge>;
  setChargesList(value: Array<Charge>): void;
  addCharges(value?: Charge, index?: number): Charge;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Charges.AsObject;
  static toObject(includeInstance: boolean, msg: Charges): Charges.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Charges, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Charges;
  static deserializeBinaryFromReader(message: Charges, reader: jspb.BinaryReader): Charges;
}

export namespace Charges {
  export type AsObject = {
    chargesList: Array<Charge.AsObject>,
  }
}

export class GetAutoChargeRequest extends jspb.Message {
  getOrganizationId(): string;
  setOrganizationId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): GetAutoChargeRequest.AsObject;
  static toObject(includeInstance: boolean, msg: GetAutoChargeRequest): GetAutoChargeRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: GetAutoChargeRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): GetAutoChargeRequest;
  static deserializeBinaryFromReader(message: GetAutoChargeRequest, reader: jspb.BinaryReader): GetAutoChargeRequest;
}

export namespace GetAutoChargeRequest {
  export type AsObject = {
    organizationId: string,
  }
}

export class AutoChargeIntent extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  getPaymentLink(): string;
  setPaymentLink(value: string): void;

  getCreatedAt(): number;
  setCreatedAt(value: number): void;

  getMonthlyMaximum(): number;
  setMonthlyMaximum(value: number): void;

  getMinimumValue(): number;
  setMinimumValue(value: number): void;

  getAmountCredits(): number;
  setAmountCredits(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): AutoChargeIntent.AsObject;
  static toObject(includeInstance: boolean, msg: AutoChargeIntent): AutoChargeIntent.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: AutoChargeIntent, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): AutoChargeIntent;
  static deserializeBinaryFromReader(message: AutoChargeIntent, reader: jspb.BinaryReader): AutoChargeIntent;
}

export namespace AutoChargeIntent {
  export type AsObject = {
    id: string,
    paymentLink: string,
    createdAt: number,
    monthlyMaximum: number,
    minimumValue: number,
    amountCredits: number,
  }
}

export class UpdateUserInfoRequest extends jspb.Message {
  hasEmail(): boolean;
  clearEmail(): void;
  getEmail(): string;
  setEmail(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): UpdateUserInfoRequest.AsObject;
  static toObject(includeInstance: boolean, msg: UpdateUserInfoRequest): UpdateUserInfoRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: UpdateUserInfoRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): UpdateUserInfoRequest;
  static deserializeBinaryFromReader(message: UpdateUserInfoRequest, reader: jspb.BinaryReader): UpdateUserInfoRequest;
}

export namespace UpdateUserInfoRequest {
  export type AsObject = {
    email: string,
  }
}

export class UserPasswordChangeTicket extends jspb.Message {
  getTicket(): string;
  setTicket(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): UserPasswordChangeTicket.AsObject;
  static toObject(includeInstance: boolean, msg: UserPasswordChangeTicket): UserPasswordChangeTicket.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: UserPasswordChangeTicket, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): UserPasswordChangeTicket;
  static deserializeBinaryFromReader(message: UserPasswordChangeTicket, reader: jspb.BinaryReader): UserPasswordChangeTicket;
}

export namespace UserPasswordChangeTicket {
  export type AsObject = {
    ticket: string,
  }
}

export interface OrganizationRoleMap {
  MEMBER: 0;
  ACCOUNTANT: 1;
  OWNER: 2;
}

export const OrganizationRole: OrganizationRoleMap;

