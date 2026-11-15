// Reading the trial brief: the free model list, the prompt list, and the catalog.
//
// The brief (the file named by MAKEFUN_PROMPT_FILE) is prose with two JSON documents
// embedded in it. The prose is decoration; this module locates each document by the
// `schema` field it carries, under this contest's prefix (`free-model-aggregator-02/`).
//
//   - the model list (`free-model-aggregator-02/models/v1`): one row per model, keyed by
//     the id the gateway dials (`openrouter/<vendor>/<model>:free`), carrying the model's
//     public rating in milli-points, its capabilities (tools, vision, JSON mode, context
//     length) and its free flag. Only free models may answer a prompt; a paid model is
//     never routed.
//   - the prompt list (`free-model-aggregator-02/prompts/v1`): the prompts the house's
//     client will send, in order, each with an id, an api (`openai` or `anthropic`),
//     whether it streams, its weight (1 or 2) and the requirements a model must satisfy
//     to serve it. The client sends the bodies; the entry routes each arrival by reading
//     the body itself, so the prompt list is mostly documentation — but it is still read,
//     because a brief without it is malformed.
//
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or
// `manifest` (the sibling -01 contest's names), and a single document may carry both
// lists; a document is identified by the arrays it holds, never by a name alone. The
// brief's prose is never parsed.
//
// Nothing here touches the clock or randomness, so `node main.ts --list` and the
// register body produce the same document, byte for byte, on a live pass and on its
// replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The prefix every schema in this contest's brief carries; brief JSON is located by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The provider's public endpoint for the free models: where the gateway reaches them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable the entrant's OpenRouter key is pushed under (an entry secret). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs from a model: a model lacking it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this module reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, taken from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * Every JSON object in `text` that carries a string `schema` member, in the order they
 * appear. A `{` that does not open a parseable object is skipped, and scanning resumes
 * just past it, so malformed fragments never hide a later document.
 */
function extractSchemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  let cursor = text.indexOf('{');
  while (cursor !== -1) {
    const closing = findMatchingCloseBrace(text, cursor);
    if (closing !== -1) {
      const parsed = parseSchemaObject(text.slice(cursor, closing + 1));
      if (parsed !== null) found.push(parsed);
    }
    cursor = text.indexOf('{', cursor + 1);
  }
  return found;
}

/** Parse `source` as JSON and return it when it is an object with a string `schema`. */
function parseSchemaObject(source: string): Json | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch {
    return null;
  }
  if (!isRecord(parsed) || typeof parsed.schema !== 'string') return null;
  return parsed;
}

/**
 * The index of the `}` that closes the object opening at `start`, or -1 when the text
 * ends first. String contents are skipped, including escaped characters inside them, so
 * braces inside a string never disturb the depth count.
 */
function findMatchingCloseBrace(text: string, start: number): number {
  let depth = 0;
  let insideString = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (insideString) {
      switch (char) {
        case '\\':
          index += 1;
          break;
        case '"':
          insideString = false;
          break;
        default:
          break;
      }
      continue;
    }
    switch (char) {
      case '"':
        insideString = true;
        break;
      case '{':
        depth += 1;
        break;
      case '}': {
        depth -= 1;
        if (depth === 0) return index;
        break;
      }
      default:
        break;
    }
  }
  return -1;
}

function asStringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asFiniteNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBooleanOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/**
 * The first document under this contest's prefix that holds one of `keys` as an array,
 * together with the key it was found under. A brief that spells its lists in the
 * sibling contest's names still reads, because the search tries every spelling.
 */
function findDocumentWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asStringOrEmpty(doc.schema).includes(SCHEMA_PREFIX)) continue;
    const key = keys.find((candidate) => Array.isArray(doc[candidate]));
    if (key !== undefined) return { doc, key };
  }
  return null;
}

/**
 * Resolve a model row's dialable id. The row gives its `id`, or its `model` under the
 * provider that names it; a row that gives the bare OpenRouter id (`vendor/model:free`)
 * and names `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`).
 */
function resolveModelId(row: Json): string {
  const raw = asStringOrEmpty(row.id) !== '' ? asStringOrEmpty(row.id) : asStringOrEmpty(row.model);
  if (raw === '') return '';
  const named = asStringOrEmpty(row.provider) !== '' ? asStringOrEmpty(row.provider) : asStringOrEmpty(row.wire);
  if (named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')) return `openrouter/${raw}`;
  return raw;
}

/**
 * One model row. The brief lists free models, so a row is free unless it says
 * `free: false`; a paid row is never routed. The display name falls back to the raw
 * id, and the rating falls back to 0 when the brief omits it.
 */
function parseModelRow(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const id = resolveModelId(value);
  if (id === '') return null;
  const named = asStringOrEmpty(value.provider) !== '' ? asStringOrEmpty(value.provider) : asStringOrEmpty(value.wire);
  const raw = asStringOrEmpty(value.id) !== '' ? asStringOrEmpty(value.id) : asStringOrEmpty(value.model);
  const displayName = asStringOrEmpty(value.display_name);
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: displayName !== '' ? displayName : raw,
    rating_milli: asFiniteNumberOrNull(value.rating_milli) ?? 0,
    context_length: asFiniteNumberOrNull(value.context_length),
    tools: asBooleanOrNull(value.tools),
    vision: asBooleanOrNull(value.vision),
    json: asBooleanOrNull(value.json),
    free: value.free !== false,
  };
}

/** The requirements a prompt row declares, keeping only the flags it sets to true. */
function parseRequirementBlock(value: unknown): Requirement {
  if (!isRecord(value)) return {};
  const requires: Requirement = {};
  if (value.tools === true) requires.tools = true;
  if (value.vision === true) requires.vision = true;
  if (value.json === true) requires.json = true;
  const minContext = asFiniteNumberOrNull(value.min_context);
  if (minContext !== null) requires.min_context = minContext;
  return requires;
}

function parsePromptRow(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asStringOrEmpty(value.id);
  if (id === '') return null;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: parseRequirementBlock(value.requires),
  };
}

/** Parse every row of `rows` with `parse`, keeping the ones that parse. */
function parseRows<T>(rows: unknown[], parse: (value: unknown) => T | null): T[] {
  const parsed: T[] = [];
  for (const row of rows) {
    const item = parse(row);
    if (item !== null) parsed.push(item);
  }
  return parsed;
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = extractSchemaDocuments(text);
  const modelsDoc = findDocumentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = parseRows(modelsDoc.doc[modelsDoc.key] as unknown[], parseModelRow);
  const promptsDoc = findDocumentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = parseRows(promptsDoc.doc[promptsDoc.key] as unknown[], parsePromptRow);
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  const free: ModelRow[] = [];
  for (const model of brief.models) {
    if (model.free) free.push(model);
  }
  return free;
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const checks: boolean[] = [
    requires.tools !== true || caps.tools === true,
    requires.vision !== true || caps.vision === true,
    requires.json !== true || caps.json === true,
    requires.min_context === undefined || (caps.context_length !== null && caps.context_length >= requires.min_context),
  ];
  return checks.every((passed) => passed);
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local
 * address: the gateway's origin differs between a live pass and its replay, and the
 * catalog is part of the state export a replay must reproduce byte for byte.
 *
 * This entry lists every free model in the brief — exactly the fourteen dialable ids
 * the catalog column scores — so keep it that way: register this set and keep GET
 * /v1/models (both auth headers) exactly that set, since listing fewer loses the
 * column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  const provider: CatalogProvider = {
    id: 'openrouter',
    base_url: OPENROUTER_PUBLIC_BASE,
    key_env: OPENROUTER_KEY_ENV,
    wire: 'openai',
    models,
  };
  return { providers: [provider] };
}
