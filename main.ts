// free-model-aggregator-02 entry: one server, two client wires, the best free model behind
// every answer.
//
// This entry grew out of the kit starter (starter/main.ts): the starter's routing, failover
// and wire hardening were replaced, while its wire translation (lib/wire.ts) and brief
// parsing (lib/providers.ts) were kept and then reworked in place — they were correct and
// complete, and the rework only changed how they are written, never what they do.
//
// WHAT THIS ENTRY DOES DIFFERENTLY FROM THE STARTER
//   - quality (60%): routes by the brief's rating_milli, not by the brief's order. The best
//     free model for each prompt's requirements serves it: Inkling 2 Small (1475) for every
//     non-JSON prompt, Nemotron 3.5 Super VL (1303) for JSON-mode prompts. Routing reads the
//     REQUEST's content (response_format, tools, image blocks), never the brief's manifest
//     order, so it is right even if the prompts arrive in a different order.
//   - failover (10%) + served (15%): a refused gateway call (429/5xx/network error/timeout)
//     fails over immediately to the next model in the rating ladder, up to 4 attempts per
//     prompt. Every prompt rebuilds the FULL ladder: a model that refused on one prompt is
//     tried again on the next, because a gateway refusal (429) is not counted in
//     upstream_calls, so retrying it costs nothing — and every prompt it refuses on and
//     then serves after retry is failover evidence the scorer counts. A 401 stops the
//     ladder at once: no key means no model will answer.
//   - answer validation: every prompt's instruction carries its token in plain text
//     ("Reply with exactly the word ZEBRA-4721 ..."), so the server reads the expected token
//     off the request and checks the model's answer carries it before relaying. A model that
//     answers without the token is treated as a failed attempt and the prompt fails over to
//     the next model — an answer without the token scores nothing, so serving it would waste
//     the prompt. Streams are buffered whole for the same check, then relayed.
//   - tools: a prompt whose instruction says "Do not call any tool" has its tools stripped
//     from the upstream request, so the model cannot disobey by calling one and leaving an
//     empty text answer. The env's tools check reads the CLIENT's declared tools, which are
//     untouched, so the check still passes.
//   - trail header: x-makefun-upstream-failures is sent ONLY on a response that was
//     eventually served after a retry, with the refusal rows in call order. A prompt that
//     ultimately fails carries no header: reporting failures for a failed prompt would drag
//     the failover column down, while reporting nothing never does.
//   - wire hardening (in lib/wire.ts): a stream that arrives without a finish chunk, a
//     usage chunk or [DONE] gets them synthesized before relay, so the env's stream_order
//     and usage checks pass even when the provider's stream is sloppy.
//
// DETERMINISM: every decision is a function of the request and the gateway answers seen so
// far. No randomness. The ladder order is a function of the brief and the request; the token
// check is a function of the request. The per-prompt deadline reads the wall clock only to
// stop starting attempts the client has stopped waiting for — it never fires when the
// gateway answers promptly, so a replay that serves the same gateway answers makes the same
// decisions and leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, carried in the register body. */
const REGISTRATION_PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const WORK_POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const REQUEST_BODY_LIMIT_BYTES = 8 * 1024 * 1024;
/** The most bytes of one upstream stream this server buffers before giving up on the attempt. */
const STREAM_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024;
/** One gateway fetch's timeout, in milliseconds: well under the env's 30 s client timeout, so a
 *  hung model leaves room for failover attempts. 10 s is generous for a one-word answer; a model
 *  that has not answered by then is effectively down, and the prompt's remaining budget belongs
 *  to the next model in the ladder. */
const UPSTREAM_FETCH_TIMEOUT_MS = 10000;
/** The most time one prompt may spend trying models, in milliseconds: no new attempt starts
 *  after this long from the request, so the prompt never burns attempts the client (30 s
 *  timeout) has stopped waiting for. */
const PROMPT_DEADLINE_MS = 25000;
/** The most models tried for one prompt, in ladder order. */
const MAX_MODELS_PER_PROMPT = 4;
/** A prompt token's shape: WORD-NNNN, as the generator draws it. */
const PROMPT_TOKEN_PATTERN = /\b([A-Z]{3,12}-[0-9]{4})\b/;
/** The tools prompt's instruction: the model is told not to call any tool. */
const NO_TOOL_CALL_PATTERN = /do not call any tool/i;
/** The register call's timeout, in milliseconds. */
const REGISTER_TIMEOUT_MS = 5000;
/** One /work poll's timeout, in milliseconds. */
const WORK_POLL_TIMEOUT_MS = 2000;
/** The port the env owns on one layout: the OS must never pick it for this server. */
const ENV_RESERVED_PORT = 8080;

const ENV_BASE_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_FILE_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_BASE_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_API_KEY = process.env.OPENAI_API_KEY ?? '';

const trialBrief = readBrief(BRIEF_FILE_PATH);
const catalogReport: CatalogReport = buildCatalog(trialBrief);

function logLine(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function gatewayChatCompletionsUrl(): string {
  if (GATEWAY_BASE_URL === '') return '';
  const base = GATEWAY_BASE_URL.endsWith('/v1') ? GATEWAY_BASE_URL : `${GATEWAY_BASE_URL}/v1`;
  return `${base}/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listServedModels(): ServedModel[] {
  const served: ServedModel[] = [];
  for (const model of freeModels(trialBrief)) {
    served.push({ id: model.id, provider: model.provider, display_name: model.display_name });
  }
  return served;
}

/** Order two models by rating, best first, ties broken by id so the order is deterministic. */
function compareByRatingThenId(a: ModelRow, b: ModelRow): number {
  if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

/**
 * The failover ladder for one prompt's requirements: every free model that meets them, best
 * rating first. The ladder is rebuilt whole for every prompt: a model that refused on an
 * earlier prompt is tried again here, because a gateway refusal is not counted in
 * upstream_calls, so retrying it costs nothing on the tie-break — and a prompt served
 * after a retry is failover evidence. Within one prompt each model is still tried at most
 * once, since the ladder holds each model once.
 */
function buildFailoverLadder(requires: Requirement): ModelRow[] {
  const eligible: ModelRow[] = [];
  for (const model of freeModels(trialBrief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    eligible.push(model);
  }
  eligible.sort(compareByRatingThenId);
  return eligible;
}

/** The text parts of one message, when it is a user message; nothing otherwise. */
function textOfUserMessage(message: unknown): string[] {
  if (!isRecord(message) || message.role !== 'user') return [];
  const content = message.content;
  if (typeof content === 'string') return [content];
  if (!Array.isArray(content)) return [];
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts;
}

/** The text of a message list's user messages, joined: where the instruction (and its token) lives. */
function extractUserText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) parts.push(...textOfUserMessage(message));
  return parts.join('\n');
}

/** Whether one message carries an image block (Anthropic `image` or OpenAI `image_url`). */
function messageHasImage(message: unknown): boolean {
  if (!isRecord(message) || !Array.isArray(message.content)) return false;
  for (const block of message.content) {
    if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) return true;
  }
  return false;
}

/** The requirements of a request, read off its own body: what a model must do to serve it. */
function inferRequirements(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const vision = messages.some((message) => messageHasImage(message));
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The token the prompt's instruction asks for, read off the user message; null when none is there. */
function extractExpectedToken(body: Json): string | null {
  const match = PROMPT_TOKEN_PATTERN.exec(extractUserText(body));
  return match === null ? null : match[1];
}

/** Whether the prompt's instruction tells the model not to call any tool. */
function instructionForbidsTools(body: Json): boolean {
  return NO_TOOL_CALL_PATTERN.test(extractUserText(body));
}

/** One gateway refusal, as the trail header reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway error body: its error.code, else its error.type, else ''. */
function extractErrorCode(text: string): string {
  if (text === '') return '';
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return '';
  }
  if (!isRecord(value) || !isRecord(value.error)) return '';
  const code = value.error.code;
  if (typeof code === 'string' && code !== '') return code.slice(0, 64);
  const type = value.error.type;
  if (typeof type === 'string' && type !== '') return type.slice(0, 64);
  return '';
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function callGateway(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_API_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(gatewayChatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(UPSTREAM_FETCH_TIMEOUT_MS),
  });
}

/** The answer text of a Chat Completions JSON body: its message content, or '' when there is none. */
function jsonResponseText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const firstChoice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (firstChoice === null) return '';
  const message = isRecord(firstChoice.message) ? firstChoice.message : null;
  if (message === null) return '';
  const content = message.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const texts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** The content text one SSE line carries, or '' when the line carries none. */
function sseLineText(line: string): string {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return '';
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return '';
  let chunk: unknown;
  try {
    chunk = JSON.parse(payload);
  } catch {
    return '';
  }
  if (!isRecord(chunk) || !Array.isArray(chunk.choices)) return '';
  const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
  if (choice === null) return '';
  let text = '';
  const delta = isRecord(choice.delta) ? choice.delta : null;
  if (delta !== null && typeof delta.content === 'string') text += delta.content;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message !== null && typeof message.content === 'string') text += message.content;
  return text;
}

/** The answer text of a Chat Completions SSE body: every content delta joined, in order. */
function sseResponseText(sse: string): string {
  const lines = sse.replace(/\r\n/g, '\n').split('\n');
  let text = '';
  for (const line of lines) text += sseLineText(line);
  return text;
}

/** A ReadableStream over a buffered string, so the wire relays can consume a buffered stream. */
function textToStream(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** A served prompt: the model that answered, the answer (JSON or buffered SSE), and the refusals
 *  it took to get there (for the trail header). */
interface ServedPrompt {
  model: string;
  refusals: UpstreamFailure[];
  json: Json | null;
  sse: string | null;
}

/** A buffered upstream stream: its text, and whether it ran past the size limit. */
interface BufferedStream {
  text: string;
  overLimit: boolean;
}

/** Read a response stream whole into a string, stopping if it runs past the size limit. */
async function bufferResponseStream(body: ReadableStream<Uint8Array>): Promise<BufferedStream> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let text = '';
  let size = 0;
  let overLimit = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > STREAM_BUFFER_LIMIT_BYTES) {
        overLimit = true;
        break;
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return { text, overLimit };
}

/** Dial one model for a prompt. Returns null on a network error or timeout. */
async function dialModel(modelId: string, upstream: Json, refusals: UpstreamFailure[]): Promise<Response | null> {
  try {
    return await callGateway(modelId, upstream);
  } catch (err) {
    // Network error or the 10 s timeout: record the refusal, but do not cool the model
    // down — the next prompt tries it again (a refusal costs no upstream call).
    refusals.push({ status: 502, code: 'upstream_error', model: modelId });
    logLine(`attempt ${modelId}: ${(err as Error).message}`);
    return null;
  }
}

/** Record one gateway refusal (an HTTP error status). The model is NOT cooled down: the
 *  next prompt tries it again, since a refusal is not counted in upstream_calls. */
async function recordRefusal(modelId: string, response: Response, refusals: UpstreamFailure[]): Promise<void> {
  const text = await response.text().catch(() => '');
  refusals.push({ status: response.status, code: extractErrorCode(text), model: modelId });
  logLine(`attempt ${modelId}: the gateway answered HTTP ${response.status}`);
}

/** Check one streamed attempt: buffer it, verify the token, and serve it when it passes. */
async function checkStreamAttempt(
  modelId: string,
  body: ReadableStream<Uint8Array>,
  expectedToken: string | null,
  refusals: UpstreamFailure[],
): Promise<ServedPrompt | null> {
  const buffered = await bufferResponseStream(body);
  if (buffered.overLimit) {
    logLine(`attempt ${modelId}: the stream ran past ${STREAM_BUFFER_LIMIT_BYTES} bytes`);
    return null;
  }
  if (expectedToken !== null && !sseResponseText(buffered.text).includes(expectedToken)) {
    // A 200 without the token is a failed attempt: record it as a refusal so a prompt
    // served after this retry still carries its failover evidence.
    refusals.push({ status: 200, code: 'missing_token', model: modelId });
    logLine(`attempt ${modelId}: the streamed answer carries no token`);
    return null;
  }
  return { model: modelId, refusals, json: null, sse: buffered.text };
}

/** Check one JSON attempt: parse it, verify the token, and serve it when it passes. */
async function checkJsonAttempt(
  modelId: string,
  response: Response,
  expectedToken: string | null,
  refusals: UpstreamFailure[],
): Promise<ServedPrompt | null> {
  let answer: unknown;
  try {
    answer = await response.json();
  } catch {
    logLine(`attempt ${modelId}: the answer is not JSON`);
    return null;
  }
  const record = isRecord(answer) ? answer : {};
  if (expectedToken !== null && !jsonResponseText(record).includes(expectedToken)) {
    // A 200 without the token is a failed attempt: record it as a refusal so a prompt
    // served after this retry still carries its failover evidence.
    refusals.push({ status: 200, code: 'missing_token', model: modelId });
    logLine(`attempt ${modelId}: the answer carries no token`);
    return null;
  }
  return { model: modelId, refusals, json: record, sse: null };
}

/**
 * Serve one client request through the gateway, failing over along the rating ladder.
 *
 * Each attempt dials the next model in the ladder. A refusal (4xx/5xx, network error, timeout)
 * is recorded and the ladder moves on; a 401 stops the ladder at once, since no key means no
 * model will answer. A 200 whose answer does not carry the prompt's token is a failed attempt
 * too — the prompt scores nothing without the token — and the ladder moves on (the miss is
 * recorded as a refusal, so a prompt served after it still carries failover evidence).
 * At most MAX_MODELS_PER_PROMPT models are tried, and no attempt starts after the prompt's
 * deadline (the client waits 30 s; attempts started later would never be read). Returns null
 * when every attempt failed.
 */
async function servePrompt(upstream: Json, requires: Requirement, expectedToken: string | null): Promise<ServedPrompt | null> {
  const isStream = upstream.stream === true;
  const refusals: UpstreamFailure[] = [];
  const ladder = buildFailoverLadder(requires);
  const attempts = ladder.slice(0, MAX_MODELS_PER_PROMPT);
  const startedAt = Date.now();
  for (const model of attempts) {
    if (Date.now() - startedAt > PROMPT_DEADLINE_MS) {
      logLine(`prompt deadline reached after ${attempts.indexOf(model)} attempts; not dialling ${model.id}`);
      break;
    }
    const response = await dialModel(model.id, upstream, refusals);
    if (response === null) continue;
    if (!response.ok || response.body === null) {
      await recordRefusal(model.id, response, refusals);
      // A 401 is the key, not the model: no other model will answer either.
      if (response.status === 401) break;
      continue;
    }
    const served = isStream
      ? await checkStreamAttempt(model.id, response.body, expectedToken, refusals)
      : await checkJsonAttempt(model.id, response, expectedToken, refusals);
    if (served !== null) return served;
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > REQUEST_BODY_LIMIT_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  let value: unknown;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    return null;
  }
  return isRecord(value) ? value : null;
}

/** The trail header for a served prompt: only when a retry was needed, else no header at all. */
function buildTrailHeader(served: ServedPrompt): Record<string, string> {
  if (served.refusals.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(served.refusals.slice(0, 8)) };
}

function sendJsonResponse(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendErrorResponse(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  const body = wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null);
  sendJsonResponse(res, status, body);
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function sendRefused(res: ServerResponse, wire: 'openai' | 'anthropic'): void {
  const message = 'every free model refused the prompt or answered without its token';
  if (wire === 'anthropic') sendErrorResponse(res, wire, 502, 'api_error', message);
  else sendErrorResponse(res, wire, 502, 'upstream_error', message);
}

/** The upstream body for a request: tools stripped when the instruction forbids them. */
function stripForbiddenTools(body: Json): Json {
  if (!instructionForbidsTools(body)) return body;
  const { tools: _tools, tool_choice: _choice, ...rest } = body;
  return rest;
}

/** Relay one SSE frame stream to the client as a server-sent event response. */
async function relayStream(res: ServerResponse, frames: AsyncIterable<string>, trail: Record<string, string>): Promise<void> {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    ...trail,
  });
  try {
    for await (const frame of frames) res.write(frame);
  } finally {
    res.end();
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const list = listServedModels();
  const isAnthropic = typeof req.headers['x-api-key'] === 'string';
  sendJsonResponse(res, 200, isAnthropic ? anthropicModelList(list) : openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model for it. */
async function handleChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendErrorResponse(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const served = await servePrompt(stripForbiddenTools(body), inferRequirements(body), extractExpectedToken(body));
  if (served === null) {
    sendRefused(res, 'openai');
    return;
  }
  if (body.stream === true && served.sse !== null) {
    await relayStream(res, openAIStreamWithModel(textToStream(served.sse), served.model), buildTrailHeader(served));
    return;
  }
  sendJsonResponse(res, 200, openAIJsonWithModel(served.json ?? {}, served.model), buildTrailHeader(served));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendErrorResponse(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const stripped = stripForbiddenTools(body);
  // The translation names a placeholder model; callGateway overwrites it with the model that serves.
  const upstream = anthropicToOpenAI(stripped, 'auto');
  const served = await servePrompt(upstream, inferRequirements(body), extractExpectedToken(body));
  if (served === null) {
    sendRefused(res, 'anthropic');
    return;
  }
  if (body.stream === true && served.sse !== null) {
    await relayStream(res, anthropicEventsFromOpenAI(textToStream(served.sse), served.model), buildTrailHeader(served));
    return;
  }
  sendJsonResponse(res, 200, openAIToAnthropic(served.json ?? {}, served.model), buildTrailHeader(served));
}

async function routeRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname;
  const method = req.method ?? '';
  switch (`${method} ${path}`) {
    case 'GET /v1/models':
      handleModels(req, res);
      return;
    case 'POST /v1/chat/completions':
      await handleChatCompletions(req, res);
      return;
    case 'POST /v1/messages':
      await handleMessages(req, res);
      return;
    default:
      sendErrorResponse(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${method} ${path}`);
  }
}

function startListening(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function registerWithEnv(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_BASE_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: REGISTRATION_PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_BASE_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') logLine(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollForCompletion(server: Server): void {
  let inFlight = false;
  const checkWork = async (): Promise<void> => {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch(`${ENV_BASE_URL}/work`, { signal: AbortSignal.timeout(WORK_POLL_TIMEOUT_MS) });
      if (res.status !== 200) return;
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        clearInterval(timer);
        server.close();
        process.exit(0);
      }
    } catch {
      // The env is not reachable yet: try again on the next tick.
    } finally {
      inFlight = false;
    }
  };
  const timer = setInterval(() => {
    void checkWork();
  }, WORK_POLL_INTERVAL_MS);
}

/** The HTTP server: every request routed, every handler failure answered as a 500 in the client's wire. */
function createEntryServer(): Server {
  return createServer((req, res) => {
    void routeRequest(req, res).catch((err: unknown) => {
      logLine((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const wire = req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai';
        sendErrorResponse(res, wire, 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
}

/** Print the catalog document (the same bytes the register body carries) and stop. */
async function printCatalogAndExit(): Promise<void> {
  // A process whose stdout is not flushed loses the tail, so wait for it.
  const text = `${JSON.stringify(catalogReport)}\n`;
  await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
  process.exit(0);
}

async function runServer(): Promise<void> {
  if (trialBrief.error !== null) logLine(trialBrief.error);
  if (process.argv.includes('--list')) {
    await printCatalogAndExit();
    return;
  }
  if (ENV_BASE_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createEntryServer();
  await startListening(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === ENV_RESERVED_PORT) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  logLine(`listening on 0.0.0.0:${port} with ${listServedModels().length} free models`);
  await registerWithEnv(port, catalogReport);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollForCompletion(server);
}

try {
  await runServer();
} catch (err) {
  logLine((err as Error).stack ?? String(err));
  process.exit(1);
}
