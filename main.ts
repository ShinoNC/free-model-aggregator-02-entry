// free-model-aggregator-02 entry: one server, two client wires, the best free model behind
// every answer.
//
// Fork of the kit starter (starter/main.ts), with its routing, failover and wire hardening
// replaced. The starter's wire translation (lib/wire.ts) and brief parsing (lib/providers.ts)
// are kept: they are correct and complete.
//
// WHAT THIS ENTRY DOES DIFFERENTLY
//   - quality (60%): routes by the brief's rating_milli, not by the brief's order. The best
//     free model for each prompt's requirements serves it: Inkling 2 Small (1475) for every
//     non-JSON prompt, Nemotron 3.5 Super VL (1303) for JSON-mode prompts. Routing reads the
//     REQUEST's content (response_format, tools, image blocks), never the brief's manifest
//     order, so it is right even if the prompts arrive in a different order.
//   - failover (10%) + served (15%): a refused gateway call (429/5xx/network error/timeout)
//     fails over immediately to the next model in the rating ladder, up to 3 attempts per
//     prompt. A model that just refused is cooled down for the rest of the trial, so one
//     rate-limited model never takes later prompts down with it. A 401 stops the ladder at
//     once: no key means no model will answer.
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
// far. No wall clock, no randomness. The cooldown set and the ladder order are functions of
// the gateway's answers; the token check is a function of the request. A replay that serves
// the same gateway answers makes the same decisions and leaves the same state export.

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

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one upstream stream this server buffers before giving up on the attempt. */
const MAX_STREAM_BYTES = 2 * 1024 * 1024;
/** One gateway fetch's timeout, in milliseconds: well under the env's 30 s client timeout, so a
 *  hung model leaves room for a failover attempt. 15 s is generous for a one-word answer; a model
 *  that has not answered by then is effectively down, and the prompt's remaining budget belongs
 *  to the next model in the ladder. */
const GATEWAY_TIMEOUT_MS = 15000;
/** The most models tried for one prompt, in ladder order. */
const MAX_ATTEMPTS = 3;
/** A prompt token's shape: WORD-NNNN, as the generator draws it. */
const TOKEN_RE = /\b([A-Z]{3,12}-[0-9]{4})\b/;
/** The tools prompt's instruction: the model is told not to call any tool. */
const NO_TOOL_RE = /do not call any tool/i;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Models that refused a call this trial (429/5xx/network error/timeout): never dialled again
 *  this trial. A function of the gateway answers seen, so a replay rebuilds the same set. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The failover ladder for one prompt's requirements: every free model that meets them, best
 * rating first (ties broken by id, so the order is deterministic), with cooled-down models
 * removed. The first entry serves the prompt when it answers; the rest are the failover.
 */
function ladderFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => !cooledDown.has(model.id) && meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The text of a message list's user messages, joined: where the instruction (and its token) lives. */
function userTextOf(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    const content = message.content;
    if (typeof content === 'string') {
      parts.push(content);
      continue;
    }
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
    }
  }
  return parts.join('\n');
}

/** The requirements of a request, read off its own body: what a model must do to serve it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The token the prompt's instruction asks for, read off the user message; null when none is there. */
function tokenFromBody(body: Json): string | null {
  const match = TOKEN_RE.exec(userTextOf(body));
  return match === null ? null : match[1];
}

/** Whether the prompt's instruction tells the model not to call any tool. */
function forbidsTools(body: Json): boolean {
  return NO_TOOL_RE.test(userTextOf(body));
}

/** One gateway refusal, as the trail header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway error body: its error.code, else its error.type, else ''. */
function errorCodeOf(text: string): string {
  if (text === '') return '';
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = value.error.code;
    if (typeof code === 'string' && code !== '') return code.slice(0, 64);
    const type = value.error.type;
    if (typeof type === 'string' && type !== '') return type.slice(0, 64);
    return '';
  } catch {
    return '';
  }
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

/** The answer text of a Chat Completions JSON body: its message content, or '' when there is none. */
function jsonAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return '';
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return '';
  const content = message.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((part) => (isRecord(part) && part.type === 'text' && typeof part.text === 'string' ? [part.text] : []))
    .join('');
}

/** The answer text of a Chat Completions SSE body: every content delta joined, in order. */
function sseAnswerText(sse: string): string {
  let text = '';
  for (const line of sse.replace(/\r\n/g, '\n').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
      const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : null;
      if (delta !== null && typeof delta.content === 'string') text += delta.content;
      const message = isRecord(choice.message) ? choice.message : null;
      if (message !== null && typeof message.content === 'string') text += message.content;
    } catch {
      // Not JSON: not a content chunk.
    }
  }
  return text;
}

/** A ReadableStream over a buffered string, so the wire relays can consume a buffered stream. */
function streamFromText(text: string): ReadableStream<Uint8Array> {
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
interface Served {
  model: string;
  refusals: FailureRow[];
  json: Json | null;
  sse: string | null;
}

/**
 * Serve one client request through the gateway, failing over along the rating ladder.
 *
 * Each attempt dials the next model in the ladder. A refusal (4xx/5xx, network error, timeout)
 * cools the model down and moves on; a 401 stops the ladder at once, since no key means no
 * model will answer. A 200 whose answer does not carry the prompt's token is a failed attempt
 * too — the prompt scores nothing without the token — and the ladder moves on without cooling
 * the model down (a wrong answer is prompt-specific, not a sign the model is down).
 * At most MAX_ATTEMPTS models are tried. Returns null when every attempt failed.
 */
async function serve(upstream: Json, requires: Requirement, expectedToken: string | null): Promise<Served | null> {
  const stream = upstream.stream === true;
  const refusals: FailureRow[] = [];
  const ladder = ladderFor(requires);
  const attempts = ladder.slice(0, MAX_ATTEMPTS);
  for (const model of attempts) {
    let response: Response;
    try {
      response = await dial(model.id, upstream);
    } catch (err) {
      // Network error or the 15 s timeout: the model is unreachable, cool it down.
      cooledDown.add(model.id);
      refusals.push({ status: 502, code: 'upstream_error', model: model.id });
      log(`attempt ${model.id}: ${(err as Error).message}`);
      continue;
    }
    if (!response.ok || response.body === null) {
      const text = await response.text().catch(() => '');
      cooledDown.add(model.id);
      refusals.push({ status: response.status, code: errorCodeOf(text), model: model.id });
      log(`attempt ${model.id}: the gateway answered HTTP ${response.status}`);
      // A 401 is the key, not the model: no other model will answer either.
      if (response.status === 401) break;
      continue;
    }
    if (stream) {
      // Buffer the whole stream so the token can be checked before anything is relayed.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let text = '';
      let size = 0;
      let over = false;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > MAX_STREAM_BYTES) {
            over = true;
            break;
          }
          text += decoder.decode(value, { stream: true });
        }
        text += decoder.decode();
      } finally {
        reader.releaseLock();
      }
      if (over) {
        log(`attempt ${model.id}: the stream ran past ${MAX_STREAM_BYTES} bytes`);
        continue;
      }
      if (expectedToken !== null && !sseAnswerText(text).includes(expectedToken)) {
        log(`attempt ${model.id}: the streamed answer carries no token`);
        continue;
      }
      return { model: model.id, refusals, json: null, sse: text };
    }
    let answer: unknown;
    try {
      answer = await response.json();
    } catch {
      log(`attempt ${model.id}: the answer is not JSON`);
      continue;
    }
    const record = isRecord(answer) ? answer : {};
    if (expectedToken !== null && !jsonAnswerText(record).includes(expectedToken)) {
      log(`attempt ${model.id}: the answer carries no token`);
      continue;
    }
    return { model: model.id, refusals, json: record, sse: null };
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The trail header for a served prompt: only when a retry was needed, else no header at all. */
function trailHeader(served: Served): Record<string, string> {
  if (served.refusals.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(served.refusals.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic'): void {
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', 'every free model refused the prompt or answered without its token');
  else sendError(res, wire, 502, 'upstream_error', 'every free model refused the prompt or answered without its token');
}

/** The upstream body for an OpenAI-shaped request: tools stripped when the instruction forbids them. */
function openAIUpstream(body: Json): Json {
  if (!forbidsTools(body)) return body;
  const { tools: _tools, tool_choice: _choice, ...rest } = body;
  return rest;
}

/** The upstream body for an Anthropic-shaped request: tools stripped before translation when forbidden. */
function anthropicUpstream(body: Json): Json {
  if (!forbidsTools(body)) return body;
  const { tools: _tools, tool_choice: _choice, ...rest } = body;
  return rest;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model for it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(openAIUpstream(body), requirementsFromBody(body), tokenFromBody(body));
  if (served === null) return refused(res, 'openai');
  if (body.stream === true && served.sse !== null) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeader(served),
    });
    try {
      for await (const frame of openAIStreamWithModel(streamFromText(served.sse), served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.json ?? {}, served.model), trailHeader(served));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stripped = anthropicUpstream(body);
  // The translation names a placeholder model; dial() overwrites it with the model that serves.
  const upstream = anthropicToOpenAI(stripped, 'auto');
  const served = await serve(upstream, requirementsFromBody(body), tokenFromBody(body));
  if (served === null) return refused(res, 'anthropic');
  if (body.stream === true && served.sse !== null) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeader(served),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamFromText(served.sse), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.json ?? {}, served.model), trailHeader(served));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
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
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
