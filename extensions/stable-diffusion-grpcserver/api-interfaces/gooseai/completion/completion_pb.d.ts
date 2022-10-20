// package: gooseai
// file: completion.proto

import * as jspb from "google-protobuf";

export class Token extends jspb.Message {
  getText(): string;
  setText(value: string): void;

  getId(): number;
  setId(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Token.AsObject;
  static toObject(includeInstance: boolean, msg: Token): Token.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Token, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Token;
  static deserializeBinaryFromReader(message: Token, reader: jspb.BinaryReader): Token;
}

export namespace Token {
  export type AsObject = {
    text: string,
    id: number,
  }
}

export class Tokens extends jspb.Message {
  clearTokensList(): void;
  getTokensList(): Array<Token>;
  setTokensList(value: Array<Token>): void;
  addTokens(value?: Token, index?: number): Token;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Tokens.AsObject;
  static toObject(includeInstance: boolean, msg: Tokens): Tokens.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Tokens, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Tokens;
  static deserializeBinaryFromReader(message: Tokens, reader: jspb.BinaryReader): Tokens;
}

export namespace Tokens {
  export type AsObject = {
    tokensList: Array<Token.AsObject>,
  }
}

export class Prompt extends jspb.Message {
  hasText(): boolean;
  clearText(): void;
  getText(): string;
  setText(value: string): void;

  hasTokens(): boolean;
  clearTokens(): void;
  getTokens(): Tokens | undefined;
  setTokens(value?: Tokens): void;

  getPromptCase(): Prompt.PromptCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Prompt.AsObject;
  static toObject(includeInstance: boolean, msg: Prompt): Prompt.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Prompt, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Prompt;
  static deserializeBinaryFromReader(message: Prompt, reader: jspb.BinaryReader): Prompt;
}

export namespace Prompt {
  export type AsObject = {
    text: string,
    tokens?: Tokens.AsObject,
  }

  export enum PromptCase {
    PROMPT_NOT_SET = 0,
    TEXT = 1,
    TOKENS = 2,
  }
}

export class LogitBias extends jspb.Message {
  hasTokens(): boolean;
  clearTokens(): void;
  getTokens(): Tokens | undefined;
  setTokens(value?: Tokens): void;

  getBias(): number;
  setBias(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LogitBias.AsObject;
  static toObject(includeInstance: boolean, msg: LogitBias): LogitBias.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LogitBias, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LogitBias;
  static deserializeBinaryFromReader(message: LogitBias, reader: jspb.BinaryReader): LogitBias;
}

export namespace LogitBias {
  export type AsObject = {
    tokens?: Tokens.AsObject,
    bias: number,
  }
}

export class LogitBiases extends jspb.Message {
  clearBiasesList(): void;
  getBiasesList(): Array<LogitBias>;
  setBiasesList(value: Array<LogitBias>): void;
  addBiases(value?: LogitBias, index?: number): LogitBias;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LogitBiases.AsObject;
  static toObject(includeInstance: boolean, msg: LogitBiases): LogitBiases.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LogitBiases, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LogitBiases;
  static deserializeBinaryFromReader(message: LogitBiases, reader: jspb.BinaryReader): LogitBiases;
}

export namespace LogitBiases {
  export type AsObject = {
    biasesList: Array<LogitBias.AsObject>,
  }
}

export class FrequencyParams extends jspb.Message {
  hasPresencePenalty(): boolean;
  clearPresencePenalty(): void;
  getPresencePenalty(): number;
  setPresencePenalty(value: number): void;

  hasFrequencyPenalty(): boolean;
  clearFrequencyPenalty(): void;
  getFrequencyPenalty(): number;
  setFrequencyPenalty(value: number): void;

  hasRepetitionPenalty(): boolean;
  clearRepetitionPenalty(): void;
  getRepetitionPenalty(): number;
  setRepetitionPenalty(value: number): void;

  hasRepetitionPenaltySlope(): boolean;
  clearRepetitionPenaltySlope(): void;
  getRepetitionPenaltySlope(): number;
  setRepetitionPenaltySlope(value: number): void;

  hasRepetitionPenaltyRange(): boolean;
  clearRepetitionPenaltyRange(): void;
  getRepetitionPenaltyRange(): number;
  setRepetitionPenaltyRange(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): FrequencyParams.AsObject;
  static toObject(includeInstance: boolean, msg: FrequencyParams): FrequencyParams.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: FrequencyParams, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): FrequencyParams;
  static deserializeBinaryFromReader(message: FrequencyParams, reader: jspb.BinaryReader): FrequencyParams;
}

export namespace FrequencyParams {
  export type AsObject = {
    presencePenalty: number,
    frequencyPenalty: number,
    repetitionPenalty: number,
    repetitionPenaltySlope: number,
    repetitionPenaltyRange: number,
  }
}

export class SamplingParams extends jspb.Message {
  clearOrderList(): void;
  getOrderList(): Array<SamplingMethodMap[keyof SamplingMethodMap]>;
  setOrderList(value: Array<SamplingMethodMap[keyof SamplingMethodMap]>): void;
  addOrder(value: SamplingMethodMap[keyof SamplingMethodMap], index?: number): SamplingMethodMap[keyof SamplingMethodMap];

  hasTemperature(): boolean;
  clearTemperature(): void;
  getTemperature(): number;
  setTemperature(value: number): void;

  hasTopP(): boolean;
  clearTopP(): void;
  getTopP(): number;
  setTopP(value: number): void;

  hasTopK(): boolean;
  clearTopK(): void;
  getTopK(): number;
  setTopK(value: number): void;

  hasTailFreeSampling(): boolean;
  clearTailFreeSampling(): void;
  getTailFreeSampling(): number;
  setTailFreeSampling(value: number): void;

  hasTypicalP(): boolean;
  clearTypicalP(): void;
  getTypicalP(): number;
  setTypicalP(value: number): void;

  hasTopA(): boolean;
  clearTopA(): void;
  getTopA(): number;
  setTopA(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SamplingParams.AsObject;
  static toObject(includeInstance: boolean, msg: SamplingParams): SamplingParams.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SamplingParams, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SamplingParams;
  static deserializeBinaryFromReader(message: SamplingParams, reader: jspb.BinaryReader): SamplingParams;
}

export namespace SamplingParams {
  export type AsObject = {
    orderList: Array<SamplingMethodMap[keyof SamplingMethodMap]>,
    temperature: number,
    topP: number,
    topK: number,
    tailFreeSampling: number,
    typicalP: number,
    topA: number,
  }
}

export class ModelParams extends jspb.Message {
  hasSamplingParams(): boolean;
  clearSamplingParams(): void;
  getSamplingParams(): SamplingParams | undefined;
  setSamplingParams(value?: SamplingParams): void;

  hasFrequencyParams(): boolean;
  clearFrequencyParams(): void;
  getFrequencyParams(): FrequencyParams | undefined;
  setFrequencyParams(value?: FrequencyParams): void;

  hasLogitBias(): boolean;
  clearLogitBias(): void;
  getLogitBias(): LogitBiases | undefined;
  setLogitBias(value?: LogitBiases): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ModelParams.AsObject;
  static toObject(includeInstance: boolean, msg: ModelParams): ModelParams.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ModelParams, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ModelParams;
  static deserializeBinaryFromReader(message: ModelParams, reader: jspb.BinaryReader): ModelParams;
}

export namespace ModelParams {
  export type AsObject = {
    samplingParams?: SamplingParams.AsObject,
    frequencyParams?: FrequencyParams.AsObject,
    logitBias?: LogitBiases.AsObject,
  }
}

export class Echo extends jspb.Message {
  hasIndex(): boolean;
  clearIndex(): void;
  getIndex(): number;
  setIndex(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Echo.AsObject;
  static toObject(includeInstance: boolean, msg: Echo): Echo.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Echo, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Echo;
  static deserializeBinaryFromReader(message: Echo, reader: jspb.BinaryReader): Echo;
}

export namespace Echo {
  export type AsObject = {
    index: number,
  }
}

export class ModuleEmbedding extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  getKey(): string;
  setKey(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ModuleEmbedding.AsObject;
  static toObject(includeInstance: boolean, msg: ModuleEmbedding): ModuleEmbedding.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ModuleEmbedding, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ModuleEmbedding;
  static deserializeBinaryFromReader(message: ModuleEmbedding, reader: jspb.BinaryReader): ModuleEmbedding;
}

export namespace ModuleEmbedding {
  export type AsObject = {
    id: string,
    key: string,
  }
}

export class Tensor extends jspb.Message {
  getTyp(): NumTypeMap[keyof NumTypeMap];
  setTyp(value: NumTypeMap[keyof NumTypeMap]): void;

  clearDimsList(): void;
  getDimsList(): Array<number>;
  setDimsList(value: Array<number>): void;
  addDims(value: number, index?: number): number;

  getData(): Uint8Array | string;
  getData_asU8(): Uint8Array;
  getData_asB64(): string;
  setData(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Tensor.AsObject;
  static toObject(includeInstance: boolean, msg: Tensor): Tensor.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Tensor, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Tensor;
  static deserializeBinaryFromReader(message: Tensor, reader: jspb.BinaryReader): Tensor;
}

export namespace Tensor {
  export type AsObject = {
    typ: NumTypeMap[keyof NumTypeMap],
    dimsList: Array<number>,
    data: Uint8Array | string,
  }
}

export class Embedding extends jspb.Message {
  hasRaw(): boolean;
  clearRaw(): void;
  getRaw(): Tensor | undefined;
  setRaw(value?: Tensor): void;

  hasModule(): boolean;
  clearModule(): void;
  getModule(): ModuleEmbedding | undefined;
  setModule(value?: ModuleEmbedding): void;

  hasPos(): boolean;
  clearPos(): void;
  getPos(): number;
  setPos(value: number): void;

  getEmbeddingCase(): Embedding.EmbeddingCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Embedding.AsObject;
  static toObject(includeInstance: boolean, msg: Embedding): Embedding.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Embedding, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Embedding;
  static deserializeBinaryFromReader(message: Embedding, reader: jspb.BinaryReader): Embedding;
}

export namespace Embedding {
  export type AsObject = {
    raw?: Tensor.AsObject,
    module?: ModuleEmbedding.AsObject,
    pos: number,
  }

  export enum EmbeddingCase {
    EMBEDDING_NOT_SET = 0,
    RAW = 1,
    MODULE = 2,
  }
}

export class EngineParams extends jspb.Message {
  hasMaxTokens(): boolean;
  clearMaxTokens(): void;
  getMaxTokens(): number;
  setMaxTokens(value: number): void;

  hasCompletions(): boolean;
  clearCompletions(): void;
  getCompletions(): number;
  setCompletions(value: number): void;

  hasLogprobs(): boolean;
  clearLogprobs(): void;
  getLogprobs(): number;
  setLogprobs(value: number): void;

  hasEcho(): boolean;
  clearEcho(): void;
  getEcho(): Echo | undefined;
  setEcho(value?: Echo): void;

  hasBestOf(): boolean;
  clearBestOf(): void;
  getBestOf(): number;
  setBestOf(value: number): void;

  clearStopList(): void;
  getStopList(): Array<Prompt>;
  setStopList(value: Array<Prompt>): void;
  addStop(value?: Prompt, index?: number): Prompt;

  hasMinTokens(): boolean;
  clearMinTokens(): void;
  getMinTokens(): number;
  setMinTokens(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EngineParams.AsObject;
  static toObject(includeInstance: boolean, msg: EngineParams): EngineParams.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EngineParams, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EngineParams;
  static deserializeBinaryFromReader(message: EngineParams, reader: jspb.BinaryReader): EngineParams;
}

export namespace EngineParams {
  export type AsObject = {
    maxTokens: number,
    completions: number,
    logprobs: number,
    echo?: Echo.AsObject,
    bestOf: number,
    stopList: Array<Prompt.AsObject>,
    minTokens: number,
  }
}

export class RequestMeta extends jspb.Message {
  hasStreaming(): boolean;
  clearStreaming(): void;
  getStreaming(): boolean;
  setStreaming(value: boolean): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): RequestMeta.AsObject;
  static toObject(includeInstance: boolean, msg: RequestMeta): RequestMeta.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: RequestMeta, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): RequestMeta;
  static deserializeBinaryFromReader(message: RequestMeta, reader: jspb.BinaryReader): RequestMeta;
}

export namespace RequestMeta {
  export type AsObject = {
    streaming: boolean,
  }
}

export class Request extends jspb.Message {
  getEngineId(): string;
  setEngineId(value: string): void;

  clearPromptList(): void;
  getPromptList(): Array<Prompt>;
  setPromptList(value: Array<Prompt>): void;
  addPrompt(value?: Prompt, index?: number): Prompt;

  hasModelParams(): boolean;
  clearModelParams(): void;
  getModelParams(): ModelParams | undefined;
  setModelParams(value?: ModelParams): void;

  hasEngineParams(): boolean;
  clearEngineParams(): void;
  getEngineParams(): EngineParams | undefined;
  setEngineParams(value?: EngineParams): void;

  hasRequestId(): boolean;
  clearRequestId(): void;
  getRequestId(): string;
  setRequestId(value: string): void;

  clearEmbeddingsList(): void;
  getEmbeddingsList(): Array<Embedding>;
  setEmbeddingsList(value: Array<Embedding>): void;
  addEmbeddings(value?: Embedding, index?: number): Embedding;

  hasOriginReceived(): boolean;
  clearOriginReceived(): void;
  getOriginReceived(): number;
  setOriginReceived(value: number): void;

  hasMeta(): boolean;
  clearMeta(): void;
  getMeta(): RequestMeta | undefined;
  setMeta(value?: RequestMeta): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Request.AsObject;
  static toObject(includeInstance: boolean, msg: Request): Request.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Request, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Request;
  static deserializeBinaryFromReader(message: Request, reader: jspb.BinaryReader): Request;
}

export namespace Request {
  export type AsObject = {
    engineId: string,
    promptList: Array<Prompt.AsObject>,
    modelParams?: ModelParams.AsObject,
    engineParams?: EngineParams.AsObject,
    requestId: string,
    embeddingsList: Array<Embedding.AsObject>,
    originReceived: number,
    meta?: RequestMeta.AsObject,
  }
}

export class LogProb extends jspb.Message {
  hasToken(): boolean;
  clearToken(): void;
  getToken(): Token | undefined;
  setToken(value?: Token): void;

  hasLogprob(): boolean;
  clearLogprob(): void;
  getLogprob(): number;
  setLogprob(value: number): void;

  hasLogprobBefore(): boolean;
  clearLogprobBefore(): void;
  getLogprobBefore(): number;
  setLogprobBefore(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LogProb.AsObject;
  static toObject(includeInstance: boolean, msg: LogProb): LogProb.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LogProb, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LogProb;
  static deserializeBinaryFromReader(message: LogProb, reader: jspb.BinaryReader): LogProb;
}

export namespace LogProb {
  export type AsObject = {
    token?: Token.AsObject,
    logprob: number,
    logprobBefore: number,
  }
}

export class TokenLogProbs extends jspb.Message {
  clearLogprobsList(): void;
  getLogprobsList(): Array<LogProb>;
  setLogprobsList(value: Array<LogProb>): void;
  addLogprobs(value?: LogProb, index?: number): LogProb;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TokenLogProbs.AsObject;
  static toObject(includeInstance: boolean, msg: TokenLogProbs): TokenLogProbs.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: TokenLogProbs, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): TokenLogProbs;
  static deserializeBinaryFromReader(message: TokenLogProbs, reader: jspb.BinaryReader): TokenLogProbs;
}

export namespace TokenLogProbs {
  export type AsObject = {
    logprobsList: Array<LogProb.AsObject>,
  }
}

export class LogProbs extends jspb.Message {
  hasTokens(): boolean;
  clearTokens(): void;
  getTokens(): TokenLogProbs | undefined;
  setTokens(value?: TokenLogProbs): void;

  clearTextOffsetList(): void;
  getTextOffsetList(): Array<number>;
  setTextOffsetList(value: Array<number>): void;
  addTextOffset(value: number, index?: number): number;

  clearTopList(): void;
  getTopList(): Array<TokenLogProbs>;
  setTopList(value: Array<TokenLogProbs>): void;
  addTop(value?: TokenLogProbs, index?: number): TokenLogProbs;

  clearTopBeforeList(): void;
  getTopBeforeList(): Array<TokenLogProbs>;
  setTopBeforeList(value: Array<TokenLogProbs>): void;
  addTopBefore(value?: TokenLogProbs, index?: number): TokenLogProbs;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LogProbs.AsObject;
  static toObject(includeInstance: boolean, msg: LogProbs): LogProbs.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LogProbs, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LogProbs;
  static deserializeBinaryFromReader(message: LogProbs, reader: jspb.BinaryReader): LogProbs;
}

export namespace LogProbs {
  export type AsObject = {
    tokens?: TokenLogProbs.AsObject,
    textOffsetList: Array<number>,
    topList: Array<TokenLogProbs.AsObject>,
    topBeforeList: Array<TokenLogProbs.AsObject>,
  }
}

export class Completion extends jspb.Message {
  getText(): string;
  setText(value: string): void;

  getIndex(): number;
  setIndex(value: number): void;

  hasLogprobs(): boolean;
  clearLogprobs(): void;
  getLogprobs(): LogProbs | undefined;
  setLogprobs(value?: LogProbs): void;

  getFinishReason(): FinishReasonMap[keyof FinishReasonMap];
  setFinishReason(value: FinishReasonMap[keyof FinishReasonMap]): void;

  getTokenIndex(): number;
  setTokenIndex(value: number): void;

  getStarted(): number;
  setStarted(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Completion.AsObject;
  static toObject(includeInstance: boolean, msg: Completion): Completion.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Completion, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Completion;
  static deserializeBinaryFromReader(message: Completion, reader: jspb.BinaryReader): Completion;
}

export namespace Completion {
  export type AsObject = {
    text: string,
    index: number,
    logprobs?: LogProbs.AsObject,
    finishReason: FinishReasonMap[keyof FinishReasonMap],
    tokenIndex: number,
    started: number,
  }
}

export class AnswerMeta extends jspb.Message {
  hasGpuId(): boolean;
  clearGpuId(): void;
  getGpuId(): string;
  setGpuId(value: string): void;

  hasCpuId(): boolean;
  clearCpuId(): void;
  getCpuId(): string;
  setCpuId(value: string): void;

  hasNodeId(): boolean;
  clearNodeId(): void;
  getNodeId(): string;
  setNodeId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): AnswerMeta.AsObject;
  static toObject(includeInstance: boolean, msg: AnswerMeta): AnswerMeta.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: AnswerMeta, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): AnswerMeta;
  static deserializeBinaryFromReader(message: AnswerMeta, reader: jspb.BinaryReader): AnswerMeta;
}

export namespace AnswerMeta {
  export type AsObject = {
    gpuId: string,
    cpuId: string,
    nodeId: string,
  }
}

export class Answer extends jspb.Message {
  getAnswerId(): string;
  setAnswerId(value: string): void;

  getCreated(): number;
  setCreated(value: number): void;

  getModel(): string;
  setModel(value: string): void;

  clearChoicesList(): void;
  getChoicesList(): Array<Completion>;
  setChoicesList(value: Array<Completion>): void;
  addChoices(value?: Completion, index?: number): Completion;

  hasRequestId(): boolean;
  clearRequestId(): void;
  getRequestId(): string;
  setRequestId(value: string): void;

  getInferenceReceived(): number;
  setInferenceReceived(value: number): void;

  hasMeta(): boolean;
  clearMeta(): void;
  getMeta(): AnswerMeta | undefined;
  setMeta(value?: AnswerMeta): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Answer.AsObject;
  static toObject(includeInstance: boolean, msg: Answer): Answer.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Answer, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Answer;
  static deserializeBinaryFromReader(message: Answer, reader: jspb.BinaryReader): Answer;
}

export namespace Answer {
  export type AsObject = {
    answerId: string,
    created: number,
    model: string,
    choicesList: Array<Completion.AsObject>,
    requestId: string,
    inferenceReceived: number,
    meta?: AnswerMeta.AsObject,
  }
}

export interface FinishReasonMap {
  NULL: 0;
  LENGTH: 1;
  STOP: 2;
  ERROR: 3;
}

export const FinishReason: FinishReasonMap;

export interface SamplingMethodMap {
  NONE: 0;
  TEMPERATURE: 1;
  TOP_K: 2;
  TOP_P: 3;
  TFS: 4;
  TOP_A: 5;
  TYPICAL_P: 6;
}

export const SamplingMethod: SamplingMethodMap;

export interface NumTypeMap {
  FP16: 0;
  FP32: 1;
  BF16: 2;
}

export const NumType: NumTypeMap;

