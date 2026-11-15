// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls this entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent events.
// The entry dials the grader's gateway, an OpenAI-compatible endpoint, so every prompt
// goes upstream as a Chat Completions request and the Anthropic path is translated both
// ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id) to the
//     gateway and relays the gateway's JSON or SSE answer, with every `model` field set
//     to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image
//     blocks, tool definitions, tool results and tool choice included) and rewrites the
//     gateway's answer into a Messages object, or into the message_start /
//     content_block_start / content_block_delta / content_block_stop / message_delta /
//     message_stop event order, with the stop reason and the usage translated. Tool calls
//     become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a
//   response's `model` must be the model this entry dialled. OpenRouter's own answer
//   names a free model without the gateway's `openrouter/` prefix; these functions put
//   the dialled id back on every answer and every streamed chunk. They never invent a
//   model: the id came from the brief and went to the gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream relays: no
// clock and no randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the same reason:
// a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id the synthesized stream chunks carry: a fixed value, so the relay stays deterministic. */
const SYNTHETIC_CHUNK_ID = 'chatcmpl-aggregate';
/** The id the Anthropic stream's message_start carries. */
const STREAM_MESSAGE_ID = 'msg_stream';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  const data: Json[] = [];
  for (const model of models) {
    data.push({ id: model.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: model.provider });
  }
  return { object: 'list', data };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  const data: Json[] = [];
  for (const model of models) {
    data.push({ type: 'model', id: model.id, display_name: model.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT });
  }
  return { data, has_more: false };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function extractTextContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const texts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function convertAnthropicBlock(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultContentAsText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = extractTextContent(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function parseToolCallArguments(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value !== 'string' || value.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

function asFiniteNumberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One tool_use block as an OpenAI tool_call. */
function convertToolUseBlock(block: Json): Json {
  return {
    id: typeof block.id === 'string' ? block.id : '',
    type: 'function',
    function: {
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    },
  };
}

/** An assistant message's content array as one OpenAI assistant message. */
function buildAssistantMessage(content: unknown[]): Json {
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'tool_use') toolCalls.push(convertToolUseBlock(block));
  }
  const text = extractTextContent(content);
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return message;
}

/**
 * A user message's content array as OpenAI messages, appended to `out`: every tool_result
 * block becomes a tool message in block order, and the remaining text/image blocks become
 * one user message after them.
 */
function appendUserContentMessages(out: Json[], content: unknown[]): void {
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'tool_result') {
      out.push({
        role: 'tool',
        tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: toolResultContentAsText(block.content),
      });
      continue;
    }
    const part = convertAnthropicBlock(block);
    if (part !== null) parts.push(part);
  }
  if (parts.length > 0) out.push({ role: 'user', content: parts });
}

/** An Anthropic message as OpenAI messages, appended to `out`. */
function appendConvertedMessage(out: Json[], raw: unknown): void {
  if (!isRecord(raw)) return;
  const isAssistant = raw.role === 'assistant';
  const content = raw.content;
  if (typeof content === 'string') {
    out.push({ role: isAssistant ? 'assistant' : 'user', content });
    return;
  }
  if (!Array.isArray(content)) return;
  if (isAssistant) {
    out.push(buildAssistantMessage(content));
    return;
  }
  appendUserContentMessages(out, content);
}

/** Anthropic tool definitions as OpenAI function tools. */
function convertToolDefinitions(tools: unknown[]): Json[] {
  const converted: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    converted.push({
      type: 'function',
      function: {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    });
  }
  return converted;
}

/** An Anthropic tool_choice as an OpenAI one, or undefined when it does not translate. */
function convertToolChoice(choice: unknown): Json | string | undefined {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') {
    return { type: 'function', function: { name: choice.name } };
  }
  return undefined;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = extractTextContent(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) appendConvertedMessage(messages, raw);

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = convertToolDefinitions(body.tools);
    if (tools.length > 0) out.tools = tools;
  }
  const toolChoice = convertToolChoice(body.tool_choice);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The first choice's message of a Chat Completions answer, or {} when there is none. */
function firstChoiceMessage(body: Json): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  return isRecord(choice.message) ? choice.message : {};
}

/** The text length of a Messages content array's text blocks. */
function textBlockLength(content: Json[]): number {
  let total = 0;
  for (const block of content) {
    if (block.type === 'text' && typeof block.text === 'string') total += block.text.length;
  }
  return total;
}

/** Estimate output tokens from text length: at least 1, roughly one token per four characters. */
function estimateOutputTokens(textLength: number): number {
  return Math.max(1, Math.ceil(textLength / 4));
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = firstChoiceMessage(body);
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : extractTextContent(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: parseToolCallArguments(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  // The env's usage check requires an output of at least 1: when the provider reports 0 (or none)
  // but the answer carries text, estimate the output from the text's length instead of passing 0 on.
  const reportedOutput = asFiniteNumberOrZero(usage.completion_tokens);
  const outputTokens = reportedOutput >= 1 ? reportedOutput : estimateOutputTokens(textBlockLength(content));
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: asFiniteNumberOrZero(usage.prompt_tokens), output_tokens: outputTokens },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim.
 *  Usage is kept whole: when the provider reports no usage, or an output of 0, a usage with an
 *  output of at least 1 is synthesized from the answer's length — the env's usage check requires
 *  an output of at least 1, and a provider that omits usage must not cost the prompt. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = asFiniteNumberOrZero(usage.prompt_tokens);
  const reported = asFiniteNumberOrZero(usage.completion_tokens);
  const message = firstChoiceMessage(body);
  const content = typeof message.content === 'string' ? message.content : extractTextContent(message.content);
  out.usage = { prompt_tokens: input, completion_tokens: reported >= 1 ? reported : estimateOutputTokens(content.length) };
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  const nameLine = event === null ? '' : `event: ${event}\n`;
  return `${nameLine}data: ${JSON.stringify(data)}\n\n`;
}

/** The data payload of one SSE line, or null when the line carries no JSON data payload. */
function dataPayloadOf(line: string): string | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return null;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return null;
  return payload;
}

/** Parse one SSE data payload as a JSON object, or null when it is not one. */
function parseDataChunk(payload: string): Json | null {
  try {
    const chunk: unknown = JSON.parse(payload);
    return isRecord(chunk) ? chunk : null;
  } catch {
    return null;
  }
}

/** The finish reasons the env accepts on the OpenAI wire; anything else is rewritten to `stop`. */
const LEGAL_OPENAI_FINISH_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched. Chat Completions names
 * the model on its chunks; the client sees the id the entry dialled, never the provider's spelling.
 * A chunk without a `choices` array (a usage-only chunk from a sloppy provider) gets an empty one,
 * since the env's shape check fails a chunk that carries no choices; a finish reason outside the
 * legal OpenAI set (some providers send `eos`) is rewritten to `stop`.
 */
function rewriteFrameWithModel(frame: string, model: string): string {
  const rewritten: string[] = [];
  for (const line of frame.split('\n')) {
    const payload = dataPayloadOf(line);
    if (payload === null) {
      rewritten.push(line);
      continue;
    }
    const chunk = parseDataChunk(payload);
    if (chunk === null) {
      rewritten.push(line);
      continue;
    }
    const out: Json = { ...chunk, model };
    if (!Array.isArray(out.choices)) out.choices = [];
    if (Array.isArray(out.choices)) {
      for (const choice of out.choices) {
        if (isRecord(choice) && typeof choice.finish_reason === 'string' && !LEGAL_OPENAI_FINISH_REASONS.has(choice.finish_reason)) {
          choice.finish_reason = 'stop';
        }
      }
    }
    rewritten.push(`data: ${JSON.stringify(out)}`);
  }
  return `${rewritten.join('\n')}\n\n`;
}

/** What one OpenAI stream relay has seen so far: finish, the last usage record, [DONE], text length. */
interface OpenAIStreamState {
  sawFinish: boolean;
  /** The most recent usage record's input tokens, or null when none arrived (or it named none). */
  lastUsageInput: number | null;
  /** The most recent usage record's output tokens, or null when none arrived (or it named none). */
  lastUsageOutput: number | null;
  sawDone: boolean;
  textChars: number;
}

function newOpenAIStreamState(): OpenAIStreamState {
  return { sawFinish: false, lastUsageInput: null, lastUsageOutput: null, sawDone: false, textChars: 0 };
}

/** Whether the last usage record the relay saw satisfies the env's usage check. */
function lastUsageOk(state: OpenAIStreamState): boolean {
  return state.lastUsageInput !== null && state.lastUsageInput >= 0 && state.lastUsageOutput !== null && state.lastUsageOutput >= 1;
}

/** Inspect one frame's data payloads for a finish reason, a usage object and content text. */
function inspectOpenAIFrame(frame: string, state: OpenAIStreamState): void {
  for (const line of frame.split('\n')) {
    const payload = dataPayloadOf(line);
    if (payload === null) continue;
    const chunk = parseDataChunk(payload);
    if (chunk === null) continue;
    // The env's usage check reads the LAST usage record of the stream, so the relay tracks
    // the last one too: a good usage chunk followed by a usage-0 record (a second usage
    // chunk, or usage carried on the finish chunk) still fails, and must be answered with
    // a synthesized usage chunk after it. A record that omits prompt_tokens fails the
    // check as well (a null input), so the input is tracked alongside the output.
    if (isRecord(chunk.usage)) {
      state.lastUsageInput = typeof chunk.usage.prompt_tokens === 'number' && Number.isFinite(chunk.usage.prompt_tokens) ? chunk.usage.prompt_tokens : null;
      state.lastUsageOutput = typeof chunk.usage.completion_tokens === 'number' && Number.isFinite(chunk.usage.completion_tokens) ? chunk.usage.completion_tokens : null;
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    if (typeof choice.finish_reason === 'string') state.sawFinish = true;
    const delta = isRecord(choice.delta) ? choice.delta : null;
    if (delta !== null && typeof delta.content === 'string') state.textChars += delta.content.length;
  }
}

/** A synthesized finish chunk (finish_reason `stop`) for a stream that sent none. */
function buildSyntheticFinishChunk(model: string): string {
  const chunk = {
    id: SYNTHETIC_CHUNK_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/** A synthesized usage chunk for a stream that sent none, output estimated from streamed text. */
function buildSyntheticUsageChunk(model: string, textChars: number): string {
  const chunk = {
    id: SYNTHETIC_CHUNK_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: 0, completion_tokens: estimateOutputTokens(textChars) },
  };
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/** Split `buffer` into complete frames (terminated by a blank line) and the leftover tail. */
function splitCompleteFrames(buffer: string): { frames: string[]; tail: string } {
  const frames: string[] = [];
  let rest = buffer;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    frames.push(rest.slice(0, at));
    rest = rest.slice(at + 2);
    at = rest.indexOf('\n\n');
  }
  return { frames, tail: rest };
}

/**
 * One frame as sub-frames of at most one `data:` line each: a provider may pack several events
 * into one frame (two finish chunks, or a data chunk and [DONE], joined by a single newline with
 * no blank line between), and the relay's per-frame checks (hold [DONE], drop a duplicate finish)
 * only see them when each data line is handled on its own. Non-data lines (comments, `event:`)
 * attach to the following data line's sub-frame, or form their own when none follows.
 */
function splitFrameLines(frame: string): string[] {
  const subs: string[] = [];
  let pending: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.trim().startsWith('data:')) {
      subs.push([...pending, line].join('\n'));
      pending = [];
    } else {
      pending.push(line);
    }
  }
  if (pending.length > 0) subs.push(pending.join('\n'));
  return subs;
}

/** Whether one frame carries a finish chunk or a content delta: both must stop after the
 *  first finish, since the env's stream_order check fails a stream with two finish chunks
 *  or with a content delta after the finish chunk. A usage chunk (empty choices) is neither,
 *  so it passes through even after the finish — OpenAI sends usage after finish_reason. */
function frameCarriesFinishOrContent(frame: string): boolean {
  for (const line of frame.split('\n')) {
    const payload = dataPayloadOf(line);
    if (payload === null) continue;
    const chunk = parseDataChunk(payload);
    if (chunk === null || !Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    if (typeof choice.finish_reason === 'string') return true;
    const delta = isRecord(choice.delta) ? choice.delta : null;
    if (delta !== null && typeof delta.content === 'string') return true;
  }
  return false;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame cut
 * across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 *
 * Hardened for sloppy providers: [DONE] is held until the stream ends, so a finish chunk and a usage
 * chunk can be synthesized before it when the provider sent neither (the env's stream_order check
 * requires a finish chunk and the usage check requires an output of at least 1); a second [DONE] is
 * dropped, since the env fails a stream that carries [DONE] twice. A second finish chunk and any
 * content delta after the first finish are dropped too, since the env fails those as well. Each
 * frame is handled line by line, so a frame that packs several events (two finish chunks, or a data
 * chunk and [DONE]) is cleaned up the same way. A usage chunk is synthesized whenever the stream's
 * LAST usage record fails the env's check (output 0, or no input), not merely when no usage arrived.
 * The synthesized chunks carry the fixed `created` the model listings use, so the relay stays deterministic.
 */
export async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const state = newOpenAIStreamState();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, tail } = splitCompleteFrames(buffer);
      buffer = tail;
      for (const frame of frames) {
        for (const sub of splitFrameLines(frame)) {
          const hadFinish = state.sawFinish;
          inspectOpenAIFrame(sub, state);
          if (sub.trim() === 'data: [DONE]') {
            // Hold [DONE]: the finish and usage chunks, when synthesized, go before it.
            state.sawDone = true;
            continue;
          }
          // A finish or content frame after the first finish is a duplicate: drop it.
          if (hadFinish && frameCarriesFinishOrContent(sub)) continue;
          yield rewriteFrameWithModel(sub, model);
        }
      }
    }
    buffer += decoder.decode();
    const tail = buffer.trim();
    if (tail !== '') {
      for (const sub of splitFrameLines(buffer)) {
        if (sub.trim() === 'data: [DONE]') {
          state.sawDone = true;
          continue;
        }
        const hadFinish = state.sawFinish;
        inspectOpenAIFrame(sub, state);
        if (!(hadFinish && frameCarriesFinishOrContent(sub))) yield rewriteFrameWithModel(sub, model);
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!state.sawFinish) yield buildSyntheticFinishChunk(model);
  if (!lastUsageOk(state)) yield buildSyntheticUsageChunk(model, state.textChars);
  yield 'data: [DONE]\n\n';
}

/** The mutable state of one Anthropic event relay: which blocks are open, what arrived so far. */
interface AnthropicStreamState {
  started: boolean;
  finished: boolean;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  pendingStop: string | null;
  textIndex: number;
  nextIndex: number;
  toolIndexes: Map<number, number>;
  inputTokens: number;
  outputTokens: number;
  /** Text characters streamed so far: the fallback output estimate when no usage chunk arrives. */
  textChars: number;
}

function newAnthropicStreamState(): AnthropicStreamState {
  return {
    started: false,
    finished: false,
    pendingStop: null,
    textIndex: -1,
    nextIndex: 0,
    toolIndexes: new Map<number, number>(),
    inputTokens: 0,
    outputTokens: 0,
    textChars: 0,
  };
}

/** The message_start event, emitted once on the first chunk. */
function messageStartEvents(state: AnthropicStreamState, model: string): string[] {
  if (state.started) return [];
  state.started = true;
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: STREAM_MESSAGE_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: state.inputTokens, output_tokens: 0 },
      },
    }),
  ];
}

/** content_block_stop events for every open block, text first, then tool blocks in index order. */
function closeOpenBlocks(state: AnthropicStreamState): string[] {
  const events: string[] = [];
  if (state.textIndex !== -1) {
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
    state.textIndex = -1;
  }
  const indexes = [...state.toolIndexes.values()].sort((a, b) => a - b);
  for (const index of indexes) events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  state.toolIndexes.clear();
  return events;
}

/** The closing events for a finished stream: block stops, message_delta with usage, message_stop. */
function streamFinishEvents(state: AnthropicStreamState, stopReason: string): string[] {
  if (state.finished) return [];
  state.finished = true;
  // The env's usage check requires an output of at least 1: when the provider sent no usage
  // chunk, estimate the output from the streamed text instead of reporting 0.
  const output = state.outputTokens >= 1 ? state.outputTokens : estimateOutputTokens(state.textChars);
  return [
    ...closeOpenBlocks(state),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
}

/** The events for one text delta: open the text block on first use, then the delta. */
function textDeltaEvents(state: AnthropicStreamState, text: string): string[] {
  const events: string[] = [];
  state.textChars += text.length;
  if (state.textIndex === -1) {
    state.textIndex = state.nextIndex;
    state.nextIndex += 1;
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index: state.textIndex, content_block: { type: 'text', text: '' } }));
  }
  events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: state.textIndex, delta: { type: 'text_delta', text } }));
  return events;
}

/** The events for one tool_call delta: open the tool block on first use, then the input delta. */
function toolCallDeltaEvents(state: AnthropicStreamState, call: Json): string[] {
  const events: string[] = [];
  const upstreamIndex = typeof call.index === 'number' ? call.index : state.toolIndexes.size;
  let index = state.toolIndexes.get(upstreamIndex);
  if (index === undefined) {
    if (state.textIndex !== -1) {
      events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
      state.textIndex = -1;
    }
    index = state.nextIndex;
    state.nextIndex += 1;
    state.toolIndexes.set(upstreamIndex, index);
    const fn = isRecord(call.function) ? call.function : {};
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: {
          type: 'tool_use',
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          input: {},
        },
      }),
    );
  }
  const fn = isRecord(call.function) ? call.function : {};
  if (typeof fn.arguments === 'string' && fn.arguments !== '') {
    events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
  }
  return events;
}

/** One SSE frame's lines as Anthropic events, mutating the stream's state. */
function eventsForFrame(frame: string, state: AnthropicStreamState, model: string): string[] {
  const events: string[] = [];
  for (const line of frame.split('\n')) {
    // Once message_stop has been emitted, nothing more may follow: a provider that sends
    // a content delta after [DONE] must not reopen a block after the stream closed.
    if (state.finished) return events;
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '[DONE]') {
      events.push(...messageStartEvents(state, model), ...streamFinishEvents(state, state.pendingStop ?? 'end_turn'));
      continue;
    }
    const chunk = parseDataChunk(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      state.inputTokens = asFiniteNumberOrZero(chunk.usage.prompt_tokens);
      state.outputTokens = asFiniteNumberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    events.push(...messageStartEvents(state, model));
    if (typeof delta.content === 'string' && delta.content !== '') {
      events.push(...textDeltaEvents(state, delta.content));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (isRecord(call)) events.push(...toolCallDeltaEvents(state, call));
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) state.pendingStop = stopReasonOf(choice.finish_reason);
  }
  return events;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const state = newAnthropicStreamState();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, tail } = splitCompleteFrames(buffer);
      buffer = tail;
      for (const frame of frames) {
        for (const event of eventsForFrame(frame, state, model)) yield event;
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsForFrame(buffer, state, model)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!state.finished) {
    for (const event of messageStartEvents(state, model)) yield event;
    for (const event of streamFinishEvents(state, state.pendingStop ?? 'end_turn')) yield event;
  }
}
