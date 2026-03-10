/**
 * Tara Hair Quiz Worker — Cloudflare Worker (single-file, dashboard-deployable)
 *
 * Architecture:
 * - /api/question   → OpenAI Responses API + direct vector-store search + KV session continuity
 * - /api/assessment → OpenAI Responses API + hosted file_search + KV session summary
 * - /api/email      → Klaviyo profile + event tracking
 *
 * Endpoints:
 *   POST /api/question   — Adaptive scalp & hair diagnostic question (default: gpt-5-nano)
 *   POST /api/assessment — Hair & Scalp Assessment Profile (default: gpt-5.4 + file_search)
 *   POST /api/email      — Klaviyo profile + event tracking
 *
 * Core env vars:
 *   OPENAI_API_KEY                         — secret
 *   OPENAI_VECTOR_STORE_IDS                — comma-separated vector store IDs for assessment file_search
 *   OPENAI_QUESTION_VECTOR_STORE_IDS       — optional comma-separated vector store IDs for direct search on questions
 *   OPENAI_QUESTION_MODEL                  — optional, default gpt-5-nano
 *   OPENAI_ASSESSMENT_MODEL                — optional, default gpt-5.4
 *   OPENAI_QUESTION_MAX_SEARCH_RESULTS     — optional, default 1
 *   OPENAI_ASSESSMENT_MAX_SEARCH_RESULTS   — optional, default 8
 *   OPENAI_QUESTION_SEARCH_SCORE_THRESHOLD — optional, default 0.55
 *   OPENAI_QUESTION_SEARCH_FILTERS_JSON    — optional JSON string for vector-store search filters
 *   OPENAI_ASSESSMENT_SEARCH_FILTERS_JSON  — optional JSON string for file_search filters
 *   OPENAI_PROMPT_CACHE_KEY_QUESTION       — optional
 *   OPENAI_PROMPT_CACHE_KEY_ASSESSMENT     — optional
 *   OPENAI_PROMPT_CACHE_RETENTION          — optional, default "in-memory"
 *   OPENAI_SERVICE_TIER                    — optional, e.g. "default" or "priority"
 *
 * Other env vars:
 *   ANTHROPIC_API_KEY        — secret (optional fallback)
 *   KLAVIYO_API_KEY          — secret
 *   KLAVIYO_LIST_ID          — text
 *   DEFAULT_PROVIDER         — "openai" or "anthropic"
 *   ALLOWED_ORIGIN           — CORS origin
 *
 * KV Namespace binding:
 *   QUIZ_SESSIONS — session state
 */

export default {
  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }

    if (request.method !== 'POST') {
      return errorResponse('Method not allowed', 405, env);
    }

    const url = new URL(request.url);
    const path = url.pathname;

    try {
      const body = await request.json();
      const provider = body.provider || env.DEFAULT_PROVIDER || 'openai';

      switch (path) {
        case '/api/question':
          return handleQuestion(body, provider, env, ctx);
        case '/api/assessment':
          return handleAssessment(body, provider, env, ctx);
        case '/api/email':
          return handleEmail(body, env, request);
        default:
          return errorResponse('Not found', 404, env);
      }
    } catch (err) {
      console.error('Worker error:', err);
      return errorResponse(err.message || 'Internal server error', 500, env);
    }
  },
};

/* ═══════════════════════════════════════════════════════════
   CORS & Response Helpers
   ═══════════════════════════════════════════════════════════ */

function corsHeaders(env) {
  return {
    'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  };
}

function jsonResponse(body, status, env) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...corsHeaders(env) },
  });
}

function errorResponse(message, status, env) {
  return jsonResponse({ error: message }, status, env);
}

/* ═══════════════════════════════════════════════════════════
   System Prompts — Tara Scalp-First Hair Consultation
   ═══════════════════════════════════════════════════════════ */

const DIAGNOSTIC_SYSTEM_PROMPT = `You are Tara's adaptive hair and scalp diagnostic engine. Your job is to ask the single best next question to reduce uncertainty about the user's hair and scalp condition, following Tara's scalp-first philosophy.

You will receive:
- consultation_state: the full prior consultation in compact structured form
- retrieved_reference_snippets: 0-1 short excerpts from Tara's hair science / ingredient knowledge base

Use the snippets when relevant for accuracy or tone calibration. Do not repeat domains already covered unless needed to resolve ambiguity.

Assessment domains and what they map to:
- scalp_health: the scalp ecosystem — sebum balance, inflammation, microbiome health, follicle environment, sensitivity, dandruff, dryness, or excess oil. The scalp is the foundation of hair health.
- hair_fall: shedding patterns, thinning, density changes, breakage vs. true hair loss, seasonal shedding, stress-related telogen effluvium, pattern thinning.
- damage_structure: cuticle integrity, porosity, breakage mid-shaft or at ends, chemical damage from color/bleach/keratin treatments, heat damage from styling tools.
- hydration_moisture: moisture balance, dry-brittle vs. over-moisturized, natural oil distribution, humidity response, protein-moisture balance.
- texture_manageability: frizz, tangles, flatness, lack of volume, coarseness, limpness, inability to hold style. Often symptoms of underlying scalp or structural issues.
- routine_habits: washing frequency, products used, styling habits, heat tool usage, overnight care, scalp care awareness.
- environment_lifestyle: climate (humidity, dry heat, AC exposure), water quality, diet, stress levels, hormonal factors, medications.

Question strategy:
- Start broad (chief complaint, hair type) then narrow toward the specific concern mechanism.
- Prioritize the domain with the biggest diagnostic gap — the one that most changes the care recommendation if answered.
- Look for root-cause patterns: scalp issues often manifest as hair problems. Oily scalp + hair fall may indicate different treatment than dry scalp + hair fall.
- If the user reports hair fall, determine whether it is breakage (structural) or shedding (scalp/follicle) — the interventions are completely different.
- If multiple concerns are present, identify the primary driver vs. secondary symptoms.
- Always think scalp-first: even when the complaint is about the hair fiber, consider what the scalp environment may be contributing.

Rules:
- Ask exactly ONE next question per turn
- Question text must be concise and conversational — ask like a caring specialist, not a form
- 3-4 options maximum
- label = natural conversational phrase (3-8 words)
- value = snake_case identifier
- question_context = one concise sentence citing 1-2 concrete prior facts explaining why this question matters now
- micro_insight = one calm sentence that makes the emerging pattern more legible to the user
- bridge_text = one personalized sentence connecting the user's latest answer to a hair/scalp insight. Must reference something specific — their answer, a mechanism, or an emerging pattern. Examples: "That washing frequency can shift the scalp's oil balance — I want to see how your follicle environment is responding." / "Breakage at mid-lengths often traces back to how the scalp nourishes new growth — let me check the foundation." Never say "next question", "loading", "please wait", or any meta-reference to the quiz itself.
- Never use vague phrases like "based on your answers so far" or "let's explore further"
- Never mention Tara or any products
- Keep the tone warm, knowledgeable, calm, and empowering
- After 5 answered questions, set complete=true

Return only valid JSON with exactly these keys:
question_text, question_context, options, complete, micro_insight, bridge_text, diagnostic_memory

diagnostic_memory must stay compact and contain exactly:
- dominant_pattern_emerging (the leading concern hypothesis)
- covered_domains (array of domain names already explored)
- evidence_map (short JSON string mapping domain → key finding)
- biggest_gap (the domain or question that would most change the profile)
- user_frustration_theme (the user's primary emotional concern)
- solution_readiness (low | medium | high — how close the profile is to actionable)

Keep diagnostic_memory concise. Do not include long evidence paragraphs.`;

const ASSESSMENT_SYSTEM_PROMPT = `You are generating Tara's Hair & Scalp Assessment — a personalized, science-informed consultation that helps the user understand exactly what is happening with their hair and scalp, and why Tara's scalp-first botanical approach is the intelligent solution.

TARA BRAND PHILOSOPHY:
Tara takes a scalp-first approach to hair care. The scalp is where hair health begins — it's the foundation. When the scalp ecosystem is balanced, healthy hair follows naturally. Tara combines high-potency botanical extracts with advanced dermatological actives to address concerns at the source, not the surface.

CLEAN BEAUTY COMMITMENT:
Tara is sulfate-free, silicone-free, paraben-free, mineral oil-free, cruelty-free, and dermatologically tested. Formulated in our own laboratory and manufactured in Spain under EU regulations (among the strictest in the world for cosmetic safety and efficacy).

KEY INGREDIENTS (use when relevant to the user's concerns):
- Black Garlic Extract: Rich in antioxidants and sulfur compounds, strengthens hair from root to tip, supports scalp circulation
- Ceramides: Restore the natural protective barrier of both scalp and hair fiber
- Niacinamide (Vitamin B3): Soothes scalp inflammation, strengthens the skin barrier, supports follicle health
- Biotin: Supports keratin production, addresses thinning and weakness
- Salicylic Acid: Gently exfoliates the scalp, unclogs follicles, addresses flaking and buildup
- Zinc Pyrithione: Anti-fungal and anti-bacterial, balances scalp microbiome
- Botanical oils (Argan, Jojoba, Rosemary): Nourish without coating, support moisture balance

KNOWLEDGE BASE USAGE:
Your knowledge base contains Tara hair science, ingredient mechanisms, product formulations, scalp health research, and brand voice guidance.
- Consult it for scalp and hair mechanisms — use the knowledge base's explanations of scalp ecosystem, follicle health, moisture balance, and structural integrity
- Consult it for product logic — understand what each Tara product does and how its ingredients address specific concerns
- Consult it for ingredient science — cite verified botanical and dermatological mechanisms
- Consult it for Tara voice — if the knowledge base contains a more precise or on-brand way of expressing an idea, use that phrasing

EXPLAIN BEFORE NAMING:
- First describe what is happening in clear, accessible terms — what's out of balance, what the user is experiencing and why
- Then introduce specific ingredient or scientific language only if it genuinely clarifies
- Make the science feel approachable. "The natural oils your scalp produces to protect and nourish new growth" before "sebum production."
- The user should feel they understand the science, not that it's being performed on them.

YOUR OBJECTIVE:
Create a deeply personalized, science-informed assessment that moves the reader through this arc:
1. RECOGNITION — "This is exactly what's happening to me"
2. ROOT CAUSE — "Now I understand why this is happening"
3. CONTRAST — "That explains why what I've tried hasn't worked"
4. SOLUTION CLARITY — "A scalp-first approach makes sense for my situation"
5. PRODUCT FIT — "Tara is designed for exactly this"
6. ACTION — "I want to start this routine"

This must happen through clarity, not pressure. Through understanding, not hype.

PERSONALIZATION RULES:
- Every section must clearly reflect the user's actual consultation answers
- If the person mentioned hair fall, dryness, scalp issues, lack of volume, product buildup, or environmental stress, use those facts explicitly
- Do not invent facts not supported by the conversation
- Connect symptoms to root causes — help the user understand WHY their hair and scalp behave the way they do
- When relevant, explain the cycle they may be stuck in (e.g., stripping shampoo → overproduction of oil → more frequent washing → more stripping) and why it persists

REASONING ORDER:
1. Infer the dominant concern pattern from the evidence
2. Infer secondary contributors and how they interact
3. Infer concern level
4. Explain what is happening at the scalp and hair level in language that mirrors their lived experience
5. Explain why conventional products likely haven't resolved it — be specific to their concern, never generic. NEVER name competitor brands. Use category language: "conventional sulfate shampoos," "silicone-based conditioners," "surface-coating treatments."
6. Explain how Tara's scalp-first botanical approach specifically addresses their pattern — let the ingredient science explain why it fits
7. Build the care protocol — recommend honestly based on the concern level

TARA VOICE:
Sound like a knowledgeable, warm hair health specialist who combines botanical expertise with dermatological understanding.
- Warm. Knowledgeable. Empowering. Clean. Science-informed. Reassuring.
- Explain before recommending. Show understanding before suggesting solutions.
- Use clear, specific language about what's happening and why.
- Create trust through knowledge. Create confidence through clarity.
- The user should feel their concerns are deeply understood and taken seriously.
- Never sound like a pushy salesperson, a generic beauty influencer, a clinical report, or a fear-based marketing script.
- No exaggeration. No fake urgency. No beauty clichés.

ADVISOR POSTURE:
- You are a trusted hair health advisor whose recommendation happens to be Tara — because it genuinely fits this person's needs
- Make the user feel understood, then guided, then confident in taking action
- Frame continuing with conventional products as staying in the same cycle — not as fear, but as honest clarity
- Frame Tara as the deeper approach that addresses root cause at the scalp level
- The recommendation should feel like the obvious next step, not an impulse buy

PRODUCT RECOMMENDATION RULES:
- You are given a list of available_products with their handles, titles, and subtitles
- Select 1 product as the hero recommendation (hero: true) — prefer a bundle or set if one exists that matches the user's concerns
- Select 2-3 additional individual products as the routine steps (hero: false)
- Each recommendation needs a step_label (e.g., "Complete Set", "Cleanse", "Condition", "Treat") and a reasoning sentence explaining why this product fits their specific concern
- Only recommend products from the available_products list — use exact handles
- If no products are available, return an empty recommended_products array

Return ONLY a valid JSON object with exactly these keys (no markdown fences, no preamble):
{
  "concern_level": "mild | moderate | significant | elevated",
  "assessment_headline": "personalized headline summarizing their hair & scalp situation",
  "assessment_summary": "2-3 sentence personalized summary of their condition",
  "what_is_happening": "detailed explanation of what is happening with their scalp and hair",
  "key_insight": "the single most important insight connecting their symptoms to root cause",
  "why_previous_products_underperformed": "why conventional products haven't resolved their specific pattern (or null if not applicable)",
  "recommended_products": [
    { "handle": "product-handle", "hero": true, "step_label": "Complete Set", "reasoning": "why this product fits their concern" },
    { "handle": "product-handle", "hero": false, "step_label": "Cleanse", "reasoning": "why this product fits their concern" }
  ],
  "care_protocol": {
    "entry_point": "full_routine | targeted_start",
    "intensive_phase": "specific usage instructions for first 4 weeks",
    "maintenance_phase": "ongoing usage after intensive phase",
    "expected_timeline": "when they should expect to see changes",
    "personalized_note": "one sentence connecting the protocol to their specific pattern"
  },
  "how_tara_works_for_you": "explanation of how Tara's scalp-first approach addresses their specific concerns",
  "ingredient_highlights": [
    { "name": "Ingredient Name", "benefit": "how this ingredient specifically addresses their concern" }
  ],
  "purchase_nudge": "one compelling sentence that makes acting feel like the obvious next step",
  "klaviyo_properties": { "concern_level": "...", "primary_concern": "...", "concern_pattern": "..." }
}

Every field is required. Do not omit any. Do not add extra keys.`;

/* ═══════════════════════════════════════════════════════════
   JSON Schemas for Structured Output
   ═══════════════════════════════════════════════════════════ */

const QUESTION_TEXT_FORMAT = {
  type: 'json_schema',
  name: 'adaptive_question',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      question_text: { type: 'string' },
      question_context: { type: 'string' },
      options: {
        type: 'array',
        maxItems: 4,
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            label: { type: 'string' },
            value: { type: 'string' },
          },
          required: ['label', 'value'],
        },
      },
      complete: { type: 'boolean' },
      micro_insight: { type: 'string' },
      bridge_text: { type: 'string' },
      diagnostic_memory: {
        type: 'object',
        additionalProperties: false,
        properties: {
          dominant_pattern_emerging: { type: 'string' },
          covered_domains: { type: 'array', items: { type: 'string' } },
          evidence_map: { type: 'string' },
          biggest_gap: { type: 'string' },
          user_frustration_theme: { type: 'string' },
          solution_readiness: { type: 'string' },
        },
        required: [
          'dominant_pattern_emerging',
          'covered_domains',
          'evidence_map',
          'biggest_gap',
          'user_frustration_theme',
          'solution_readiness',
        ],
      },
    },
    required: [
      'question_text',
      'question_context',
      'options',
      'complete',
      'micro_insight',
      'bridge_text',
      'diagnostic_memory',
    ],
  },
};

const ASSESSMENT_TEXT_FORMAT = {
  type: 'json_schema',
  name: 'hair_scalp_assessment',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      concern_level: { type: 'string' },
      assessment_headline: { type: 'string' },
      assessment_summary: { type: 'string' },
      what_is_happening: { type: 'string' },
      key_insight: { type: 'string' },
      why_previous_products_underperformed: { anyOf: [{ type: 'string' }, { type: 'null' }] },
      recommended_products: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            handle: { type: 'string' },
            hero: { type: 'boolean' },
            step_label: { type: 'string' },
            reasoning: { type: 'string' },
          },
          required: ['handle', 'hero', 'step_label', 'reasoning'],
        },
      },
      care_protocol: {
        type: 'object',
        additionalProperties: false,
        properties: {
          entry_point: { type: 'string' },
          intensive_phase: { type: 'string' },
          maintenance_phase: { type: 'string' },
          expected_timeline: { type: 'string' },
          personalized_note: { type: 'string' },
        },
        required: ['entry_point', 'intensive_phase', 'maintenance_phase', 'expected_timeline', 'personalized_note'],
      },
      how_tara_works_for_you: { type: 'string' },
      ingredient_highlights: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: { name: { type: 'string' }, benefit: { type: 'string' } },
          required: ['name', 'benefit'],
        },
      },
      purchase_nudge: { type: 'string' },
      klaviyo_properties: {
        type: 'object',
        additionalProperties: false,
        properties: {
          concern_level: { type: 'string' },
          primary_concern: { type: 'string' },
          concern_pattern: { type: 'string' },
        },
        required: ['concern_level', 'primary_concern', 'concern_pattern'],
      },
    },
    required: [
      'concern_level', 'assessment_headline', 'assessment_summary',
      'what_is_happening', 'key_insight', 'why_previous_products_underperformed',
      'recommended_products', 'care_protocol', 'how_tara_works_for_you',
      'ingredient_highlights', 'purchase_nudge', 'klaviyo_properties',
    ],
  },
};

/* ═══════════════════════════════════════════════════════════
   KV Session State
   ═══════════════════════════════════════════════════════════ */

const SESSION_TTL = 3600;

async function loadSession(sessionId, env) {
  if (!env.QUIZ_SESSIONS || !sessionId) return null;
  const raw = await env.QUIZ_SESSIONS.get(sessionId);
  return raw ? JSON.parse(raw) : null;
}

function initSession() {
  return {
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    turn_count: 0,
    status: 'in_progress',
    quiz_answers: [],
    latest_question: null,
    last_response_id: null,
    diagnostic_memory: {
      dominant_pattern_emerging: '',
      covered_domains: [],
      evidence_map: '{}',
      biggest_gap: '',
      user_frustration_theme: '',
      solution_readiness: 'low',
    },
    narrative_state: {
      dominant_pattern_emerging: '',
      user_frustration_theme: '',
      solution_readiness: 'low',
      key_turning_points: [],
    },
  };
}

function appendTurn(session, questionText, answerText, answerMode = 'choice') {
  session.turn_count++;
  session.updated_at = new Date().toISOString();
  session.quiz_answers.push({
    turn: session.turn_count,
    question_text: questionText,
    answer_text: answerText,
    answer_mode: answerMode,
  });
}

function updateLatestQuestion(session, parsed) {
  session.latest_question = {
    question_text: parsed.question_text || '',
    question_context: parsed.question_context || '',
    options: parsed.options || [],
    micro_insight: parsed.micro_insight || '',
    bridge_text: parsed.bridge_text || '',
  };

  if (parsed.diagnostic_memory) {
    session.diagnostic_memory = {
      dominant_pattern_emerging: parsed.diagnostic_memory.dominant_pattern_emerging || '',
      covered_domains: parsed.diagnostic_memory.covered_domains || [],
      evidence_map: parsed.diagnostic_memory.evidence_map || '{}',
      biggest_gap: parsed.diagnostic_memory.biggest_gap || '',
      user_frustration_theme: parsed.diagnostic_memory.user_frustration_theme || '',
      solution_readiness: parsed.diagnostic_memory.solution_readiness || 'low',
    };

    session.narrative_state = {
      dominant_pattern_emerging: session.diagnostic_memory.dominant_pattern_emerging,
      user_frustration_theme: session.diagnostic_memory.user_frustration_theme,
      solution_readiness: session.diagnostic_memory.solution_readiness,
      key_turning_points: session.narrative_state?.key_turning_points || [],
    };
  }
}

async function saveSession(sessionId, session, env) {
  if (!env.QUIZ_SESSIONS || !sessionId) return;
  session.updated_at = new Date().toISOString();
  await env.QUIZ_SESSIONS.put(sessionId, JSON.stringify(session), {
    expirationTtl: SESSION_TTL,
  });
}

/* ═══════════════════════════════════════════════════════════
   OpenAI Responses API + Direct Vector Search
   ═══════════════════════════════════════════════════════════ */

const OPENAI_RESPONSES_URL = 'https://api.openai.com/v1/responses';
const OPENAI_VECTOR_STORES_BASE = 'https://api.openai.com/v1/vector_stores';

function getVectorStoreIds(env) {
  const raw = env.OPENAI_VECTOR_STORE_IDS || '';
  const ids = raw.split(',').map(s => s.trim()).filter(Boolean);
  if (!ids.length) {
    console.warn('OPENAI_VECTOR_STORE_IDS not configured — assessment file_search will be skipped');
  }
  return ids;
}

function getQuestionVectorStoreIds(env) {
  const raw = env.OPENAI_QUESTION_VECTOR_STORE_IDS || '';
  const ids = raw.split(',').map(s => s.trim()).filter(Boolean);
  if (ids.length > 0) return ids;
  const assessmentIds = getVectorStoreIds(env);
  return assessmentIds.length ? [assessmentIds[0]] : [];
}

function toPositiveInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function toOptionalNumber(value, fallback = null) {
  if (value === undefined || value === null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseOptionalJson(value, fallback = null) {
  if (value == null || value === '') return fallback;
  if (typeof value === 'object') return value;
  try { return JSON.parse(value); } catch { return fallback; }
}

function safeParseEvidenceMap(raw) { return parseOptionalJson(raw, {}); }

function getQuestionSearchFilters(body, env) {
  return parseOptionalJson(body.question_search_filters || env.OPENAI_QUESTION_SEARCH_FILTERS_JSON, null);
}

function getAssessmentSearchFilters(body, env) {
  return parseOptionalJson(body.assessment_search_filters || env.OPENAI_ASSESSMENT_SEARCH_FILTERS_JSON, null);
}

function getQuestionPromptCacheKey(env) {
  return env.OPENAI_PROMPT_CACHE_KEY_QUESTION || 'tara-quiz-question-v1';
}

function getAssessmentPromptCacheKey(env) {
  return env.OPENAI_PROMPT_CACHE_KEY_ASSESSMENT || 'tara-quiz-assessment-v1';
}

function normalizeQuestionSearchResult(result) {
  const content = Array.isArray(result?.content) ? result.content : [];
  const text = content
    .map(item => {
      if (typeof item?.text === 'string') return item.text;
      if (typeof item?.content === 'string') return item.content;
      return '';
    })
    .filter(Boolean).join('\n').trim();
  return {
    file_id: result?.file_id || '',
    filename: result?.filename || '',
    score: typeof result?.score === 'number' ? result.score : 0,
    text,
  };
}

function dedupeSnippets(items, maxResults) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const key = `${item.file_id || item.filename}::${item.text.slice(0, 280)}`;
    if (!item.text || seen.has(key)) continue;
    seen.add(key);
    out.push(item);
    if (out.length >= maxResults) break;
  }
  return out;
}

function buildQuestionSearchQuery(session, latestAnswer) {
  const coveredDomains = (session.diagnostic_memory?.covered_domains || []).join(', ');
  const biggestGap = session.diagnostic_memory?.biggest_gap || '';
  const dominantPattern = session.diagnostic_memory?.dominant_pattern_emerging || '';
  const frustrationTheme = session.diagnostic_memory?.user_frustration_theme || '';
  const recentTurns = (session.quiz_answers || []).slice(-2)
    .map(qa => `Q: ${qa.question_text} | A: ${qa.answer_text}`).join('\n');
  return [
    'Find the single most relevant hair science / scalp health / ingredient excerpt for selecting the next adaptive question.',
    latestAnswer ? `Latest answer: ${latestAnswer}` : '',
    recentTurns ? `Recent turns:\n${recentTurns}` : '',
    dominantPattern ? `Dominant pattern: ${dominantPattern}` : '',
    coveredDomains ? `Covered domains: ${coveredDomains}` : '',
    biggestGap ? `Biggest gap: ${biggestGap}` : '',
    frustrationTheme ? `User frustration theme: ${frustrationTheme}` : '',
  ].filter(Boolean).join('\n\n');
}

async function searchVectorStore({ apiKey, vectorStoreId, query, maxResults = 2, scoreThreshold = 0.55, filters = null }) {
  const body = { query, max_num_results: maxResults, rewrite_query: false, ranking_options: { ranker: 'none', score_threshold: scoreThreshold } };
  if (filters) body.filters = filters;
  const res = await fetch(`${OPENAI_VECTOR_STORES_BASE}/${vectorStoreId}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) { const err = await res.text(); throw new Error(`Vector store search ${res.status}: ${err}`); }
  const data = await res.json();
  return Array.isArray(data?.data) ? data.data.map(normalizeQuestionSearchResult) : [];
}

async function retrieveQuestionSnippets({ apiKey, vectorStoreIds, query, maxResults = 2, scoreThreshold = 0.55, filters = null }) {
  const validIds = vectorStoreIds.filter(Boolean);
  if (!validIds.length) return [];
  const allResults = await Promise.all(
    validIds.map(vectorStoreId =>
      searchVectorStore({ apiKey, vectorStoreId, query, maxResults, scoreThreshold, filters })
        .catch(err => { console.error(`Vector search ${vectorStoreId}:`, err); return []; })
    )
  );
  const collected = allResults.flat().sort((a, b) => (b.score || 0) - (a.score || 0));
  return dedupeSnippets(collected, maxResults);
}

async function callResponses({ model, instructions, input, tools = [], textFormat = null, textVerbosity = null, temperature = null, reasoningEffort = null, maxOutputTokens = null, previousResponseId = null, promptCacheKey = null, promptCacheRetention = null, store = true, serviceTier = null }, apiKey) {
  const body = { model, instructions, input, store, truncation: 'auto' };
  if (previousResponseId) body.previous_response_id = previousResponseId;
  if (reasoningEffort && reasoningEffort !== 'none') { body.reasoning = { effort: reasoningEffort }; }
  else if (temperature != null) { body.temperature = temperature; }
  if (maxOutputTokens) body.max_output_tokens = maxOutputTokens;
  if (tools.length) body.tools = tools;
  if (textFormat || textVerbosity) { body.text = {}; if (textFormat) body.text.format = textFormat; if (textVerbosity) body.text.verbosity = textVerbosity; }
  if (promptCacheKey) body.prompt_cache_key = promptCacheKey;
  if (promptCacheRetention) body.prompt_cache_retention = promptCacheRetention;
  if (serviceTier) body.service_tier = serviceTier;

  const startTime = Date.now();
  const res = await fetch(OPENAI_RESPONSES_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify(body),
  });
  const durationMs = Date.now() - startTime;
  if (!res.ok) { const err = await res.text(); throw new Error(`OpenAI Responses API ${res.status} (${(durationMs / 1000).toFixed(1)}s): ${err}`); }
  const data = await res.json();
  const textContent = data.output?.filter(item => item.type === 'message')?.flatMap(item => item.content)?.filter(c => c.type === 'output_text')?.map(c => c.text)?.join('') || '{}';
  const usage = data.usage || {};
  return { text: textContent, responseId: data.id, durationMs, usage, incompleteDetails: data.incomplete_details || null };
}

/* ═══════════════════════════════════════════════════════════
   Anthropic Claude API
   ═══════════════════════════════════════════════════════════ */

async function callClaude({ model, max_tokens, system, messages }, env) {
  if (!env.ANTHROPIC_API_KEY) throw new Error('ANTHROPIC_API_KEY not configured');
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': env.ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model, max_tokens, system, messages }),
  });
  if (!res.ok) { const err = await res.text(); throw new Error(`Anthropic API error ${res.status}: ${err}`); }
  return res.json();
}

/* ═══════════════════════════════════════════════════════════
   Input Builders
   ═══════════════════════════════════════════════════════════ */

function buildQuestionInput(session, latestAnswer, latestAnswerMode = 'choice', retrievedSnippets = []) {
  const payload = {
    latest_user_answer: latestAnswer,
    latest_answer_mode: latestAnswerMode,
    consultation_state: {
      answered_questions: session.turn_count,
      remaining_questions: Math.max(0, 5 - session.turn_count),
      all_turns: (session.quiz_answers || []).map(qa => ({ turn: qa.turn, question_text: qa.question_text, answer_text: qa.answer_text, answer_mode: qa.answer_mode })),
      latest_question: session.latest_question?.question_text || null,
      diagnostic_memory: {
        dominant_pattern_emerging: session.diagnostic_memory?.dominant_pattern_emerging || '',
        covered_domains: session.diagnostic_memory?.covered_domains || [],
        evidence_map: safeParseEvidenceMap(session.diagnostic_memory?.evidence_map),
        biggest_gap: session.diagnostic_memory?.biggest_gap || '',
        user_frustration_theme: session.diagnostic_memory?.user_frustration_theme || '',
        solution_readiness: session.diagnostic_memory?.solution_readiness || 'low',
      },
    },
    retrieved_reference_snippets: retrievedSnippets.map((item, i) => ({ rank: i + 1, source: item.filename || item.file_id || `snippet_${i + 1}`, relevance_score: item.score, text: item.text })),
  };
  return JSON.stringify(payload, null, 2);
}

function buildAssessmentInput(conversationHistory, products, session = null) {
  const summary = {
    total_turns: conversationHistory.length,
    evidence: conversationHistory.map((entry, i) => ({ turn: i + 1, question: entry.question || '', answer: entry.answer_label || entry.answer || '', input_type: entry.input_type || 'choice' })),
    diagnostic_memory: session?.diagnostic_memory || null,
    narrative_state: session?.narrative_state || null,
    available_products: (products || []).map(p => ({ handle: p.handle, title: p.title, subtitle: p.subtitle })),
  };
  return JSON.stringify(summary, null, 2);
}

/* ═══════════════════════════════════════════════════════════
   /api/question
   ═══════════════════════════════════════════════════════════ */

async function handleQuestion(body, provider, env, ctx) {
  const { conversation_history = [], session_id, answer } = body;
  if (!conversation_history.length && !answer) return errorResponse('conversation_history or answer is required', 400, env);

  if (provider === 'openai' && env.OPENAI_API_KEY) return handleQuestionOpenAI(body, env);
  if (env.QUIZ_SESSIONS && session_id) return handleQuestionAnthropicSession(body, env);

  let userMessage = 'Here is the consultation so far:\n\n';
  conversation_history.forEach((entry, i) => { userMessage += `Q${i + 1}: ${entry.question}\nA${i + 1}: "${entry.answer_label}"\n\n`; });
  const totalAnswered = conversation_history.length;
  const remaining = Math.max(0, 5 - totalAnswered);
  userMessage += totalAnswered >= 5 ? 'You MUST respond with {"complete": true} now.' : `${totalAnswered} of 5 answered (${remaining} remaining). Generate the next question.`;

  try {
    const response = await callClaude({ model: 'claude-haiku-4-5-20251001', max_tokens: 512, system: DIAGNOSTIC_SYSTEM_PROMPT, messages: [{ role: 'user', content: userMessage }] }, env);
    const parsed = parseQuestionJson(response.content[0]?.text || '{}');
    delete parsed.diagnostic_memory;
    return jsonResponse(parsed, 200, env);
  } catch (err) {
    console.error('Anthropic stateless question error:', err);
    const fallback = getFallbackQuestion(totalAnswered);
    delete fallback.diagnostic_memory;
    return jsonResponse(fallback, 200, env);
  }
}

async function handleQuestionOpenAI(body, env) {
  const { conversation_history = [], session_id, answer } = body;
  if (!session_id) return errorResponse('session_id is required for OpenAI quiz sessions', 400, env);

  const apiKey = env.OPENAI_API_KEY;
  let session = await loadSession(session_id, env);

  if (!session) {
    session = initSession();
    if (Array.isArray(conversation_history) && conversation_history.length) {
      for (const entry of conversation_history) {
        const q = entry?.question || '', a = entry?.answer_label || entry?.answer || '', mode = entry?.input_type || 'choice';
        if (!q && !a) continue;
        appendTurn(session, q, a, mode);
      }
      const lastEntry = conversation_history[conversation_history.length - 1];
      session.latest_question = { question_text: lastEntry?.question || '', question_context: '', options: [], micro_insight: '', bridge_text: '' };
    }
  }

  const anchor = answer || conversation_history[conversation_history.length - 1];
  const answerText = typeof anchor === 'string' ? anchor : anchor?.answer_label || anchor?.answer || '';
  const answerMode = typeof anchor === 'string' ? 'choice' : anchor?.input_type || 'choice';
  if (!answerText) return errorResponse('answer is required', 400, env);

  const currentQuestionText = session.latest_question?.question_text || (typeof anchor === 'object' ? anchor?.question : '') || "What's your biggest hair or scalp concern right now?";
  const lastRecorded = session.quiz_answers[session.quiz_answers.length - 1];
  const isDuplicateTurn = !!lastRecorded && lastRecorded.question_text === currentQuestionText && lastRecorded.answer_text === answerText;
  if (!isDuplicateTurn) appendTurn(session, currentQuestionText, answerText, answerMode);

  if (session.turn_count >= 5) {
    session.status = 'awaiting_email';
    await saveSession(session_id, session, env);
    return jsonResponse({ complete: true }, 200, env);
  }

  const questionModel = body.openai_question_model || env.OPENAI_QUESTION_MODEL || 'gpt-5-nano';
  const questionEffort = body.openai_question_effort || env.OPENAI_QUESTION_EFFORT || 'low';
  const questionVectorStoreIds = getQuestionVectorStoreIds(env);
  const maxQuestionSearchResults = toPositiveInt(body.max_search_results ?? env.OPENAI_QUESTION_MAX_SEARCH_RESULTS, 1);
  const searchScoreThreshold = toOptionalNumber(body.question_search_score_threshold ?? env.OPENAI_QUESTION_SEARCH_SCORE_THRESHOLD, 0.55);
  const searchFilters = getQuestionSearchFilters(body, env);

  let retrievedSnippets = [], searchDurationMs = 0;
  if (session.turn_count > 1 && questionVectorStoreIds.length > 0) {
    try {
      const searchStart = Date.now();
      const query = buildQuestionSearchQuery(session, answerText);
      retrievedSnippets = await retrieveQuestionSnippets({ apiKey, vectorStoreIds: questionVectorStoreIds, query, maxResults: maxQuestionSearchResults, scoreThreshold: searchScoreThreshold, filters: searchFilters });
      searchDurationMs = Date.now() - searchStart;
    } catch (retrievalErr) { console.error('Question vector search error:', retrievalErr); }
  }

  const input = buildQuestionInput(session, answerText, answerMode, retrievedSnippets);

  try {
    const { text, durationMs, usage, incompleteDetails } = await callResponses({
      model: questionModel, instructions: `${DIAGNOSTIC_SYSTEM_PROMPT}\n\nUse retrieved_reference_snippets when relevant. Return only JSON.`,
      input, tools: [], textFormat: QUESTION_TEXT_FORMAT, textVerbosity: 'low', reasoningEffort: questionEffort,
      previousResponseId: null, store: false, promptCacheKey: getQuestionPromptCacheKey(env),
      promptCacheRetention: env.OPENAI_PROMPT_CACHE_RETENTION || 'in_memory', serviceTier: env.OPENAI_SERVICE_TIER || 'default',
    }, apiKey);

    const parsed = parseQuestionJson(text);
    updateLatestQuestion(session, parsed);
    if (parsed.complete) session.status = 'awaiting_email';
    await saveSession(session_id, session, env);

    const clientResponse = { question_text: parsed.question_text, question_context: parsed.question_context, options: parsed.options, complete: parsed.complete, micro_insight: parsed.micro_insight, bridge_text: parsed.bridge_text || '' };
    if (body.debug) {
      clientResponse._debug = { provider: 'openai', mode: 'direct_vector_search_compact_context', model: questionModel, effort: questionEffort, turn: session.turn_count, search_store_ids: questionVectorStoreIds, search_results_used: retrievedSnippets.length, search_duration_ms: searchDurationMs, search_duration: `${(searchDurationMs / 1000).toFixed(1)}s`, llm_duration_ms: durationMs, llm_duration: `${(durationMs / 1000).toFixed(1)}s`, total_duration_ms: searchDurationMs + durationMs, total_duration: `${((searchDurationMs + durationMs) / 1000).toFixed(1)}s`, tokens: usage, cached_tokens: usage?.input_tokens_details?.cached_tokens || 0, parsed: !!parsed._parsed, incomplete_details: incompleteDetails };
    }
    return jsonResponse(clientResponse, 200, env);
  } catch (err) {
    console.error('OpenAI question error:', err);
    const fallback = getFallbackQuestion(session.turn_count);
    updateLatestQuestion(session, fallback);
    await saveSession(session_id, session, env);
    const clientResponse = { question_text: fallback.question_text, question_context: fallback.question_context, options: fallback.options, complete: fallback.complete, micro_insight: fallback.micro_insight, bridge_text: fallback.bridge_text || '' };
    if (body.debug) { clientResponse._debug = { error: err.message, provider: 'openai', mode: 'direct_vector_search_compact_context', model: questionModel, effort: questionEffort, turn: session.turn_count }; }
    return jsonResponse(clientResponse, 200, env);
  }
}

async function handleQuestionAnthropicSession(body, env) {
  const { conversation_history = [], session_id, answer } = body;
  let session = await loadSession(session_id, env);

  if (!session) {
    const anchor = answer || conversation_history[0];
    if (!anchor) return errorResponse('No answer for new session', 400, env);
    const answerText = typeof anchor === 'string' ? anchor : anchor.answer_label || anchor.answer || '';
    const questionText = typeof anchor === 'string' ? "What's your biggest hair or scalp concern right now?" : anchor.question || "What's your biggest hair or scalp concern right now?";
    session = initSession();
    appendTurn(session, questionText, answerText);
    session.messages = [{ role: 'user', content: `My answer to "${questionText}": ${answerText}` }];
  } else if (answer) {
    const answerText = typeof answer === 'string' ? answer : answer.answer_label || answer.answer || '';
    const questionText = session.latest_question?.question_text || '';
    appendTurn(session, questionText, answerText);
    session.messages.push({ role: 'user', content: answerText });
  }

  if (session.turn_count >= 5) {
    session.status = 'awaiting_email';
    await saveSession(session_id, session, env);
    return jsonResponse({ complete: true }, 200, env);
  }

  const latestAnswer = session.quiz_answers[session.quiz_answers.length - 1]?.answer_text || '';
  const structuredInput = buildQuestionInput(session, latestAnswer);

  try {
    const response = await callClaude({ model: 'claude-haiku-4-5-20251001', max_tokens: 768, system: DIAGNOSTIC_SYSTEM_PROMPT, messages: [{ role: 'user', content: structuredInput }] }, env);
    const text = response.content[0]?.text || '{}';
    const parsed = parseQuestionJson(text);
    session.messages.push({ role: 'assistant', content: text });
    updateLatestQuestion(session, parsed);
    if (parsed.complete) session.status = 'awaiting_email';
    await saveSession(session_id, session, env);
    const clientResponse = { ...parsed };
    delete clientResponse.diagnostic_memory;
    return jsonResponse(clientResponse, 200, env);
  } catch (err) {
    console.error('Anthropic session question error:', err);
    const fallback = getFallbackQuestion(session.turn_count);
    updateLatestQuestion(session, fallback);
    if (session_id) await saveSession(session_id, session, env);
    const clientResponse = { ...fallback };
    delete clientResponse.diagnostic_memory;
    return jsonResponse(clientResponse, 200, env);
  }
}

/* ═══════════════════════════════════════════════════════════
   /api/assessment
   ═══════════════════════════════════════════════════════════ */

async function handleAssessment(body, provider, env, ctx) {
  const { email, conversation_history = [], products = [], session_id } = body;
  if (!conversation_history.length) return errorResponse('conversation_history is required', 400, env);

  let session = null;
  if (session_id && env.QUIZ_SESSIONS) session = await loadSession(session_id, env);
  let parsed;

  if (provider === 'openai' && env.OPENAI_API_KEY) {
    const apiKey = env.OPENAI_API_KEY;
    const vectorStoreIds = getVectorStoreIds(env);
    const input = buildAssessmentInput(conversation_history, products, session);
    try {
      const assessmentModel = body.openai_assessment_model || env.OPENAI_ASSESSMENT_MODEL || 'gpt-5.4';
      const assessmentEffort = body.openai_assessment_effort || env.OPENAI_ASSESSMENT_EFFORT || 'high';
      const maxSearchResults = toPositiveInt(body.max_search_results ?? env.OPENAI_ASSESSMENT_MAX_SEARCH_RESULTS, 8);
      const assessmentSearchFilters = getAssessmentSearchFilters(body, env);
      const tools = [];
      if (vectorStoreIds.length > 0) {
        const fileSearchTool = { type: 'file_search', vector_store_ids: vectorStoreIds, max_num_results: maxSearchResults };
        if (assessmentSearchFilters) fileSearchTool.filters = assessmentSearchFilters;
        tools.push(fileSearchTool);
      }
      const { text, incompleteDetails } = await callResponses({
        model: assessmentModel, instructions: `${ASSESSMENT_SYSTEM_PROMPT}\n\nReturn only JSON.`, input, tools,
        textFormat: ASSESSMENT_TEXT_FORMAT, reasoningEffort: assessmentEffort, maxOutputTokens: 16384,
        previousResponseId: null, promptCacheKey: getAssessmentPromptCacheKey(env),
        promptCacheRetention: env.OPENAI_PROMPT_CACHE_RETENTION || 'in_memory', serviceTier: env.OPENAI_SERVICE_TIER || 'default',
      }, apiKey);
      if (incompleteDetails) console.warn('Assessment response incomplete:', JSON.stringify(incompleteDetails), '| text length:', text.length);
      parsed = parseAssessmentJson(text);
    } catch (err) {
      console.error('OpenAI assessment error:', err);
      return errorResponse('Assessment generation failed: ' + err.message, 500, env);
    }
  } else {
    try {
      const response = await callClaude({ model: 'claude-opus-4-6', max_tokens: 4096, system: ASSESSMENT_SYSTEM_PROMPT, messages: [{ role: 'user', content: buildAssessmentInput(conversation_history, products, session) }] }, env);
      parsed = parseAssessmentJson(response.content[0]?.text || '{}');
    } catch (err) {
      console.error('Anthropic assessment error:', err);
      return errorResponse(`Assessment generation failed: ${err.message}`, 500, env);
    }
  }

  if (email && parsed.klaviyo_properties && env.KLAVIYO_API_KEY) {
    upsertKlaviyoProfile(email, parsed.klaviyo_properties, env).catch(e => console.error('Klaviyo assessment update failed:', e));
  }
  delete parsed.klaviyo_properties;

  if (session && session_id) {
    session.status = 'complete';
    const assessmentSave = saveSession(session_id, session, env);
    if (ctx?.waitUntil) ctx.waitUntil(assessmentSave); else await assessmentSave;
  }

  return jsonResponse(parsed, 200, env);
}

/* ═══════════════════════════════════════════════════════════
   /api/email
   ═══════════════════════════════════════════════════════════ */

async function handleEmail(body, env, request) {
  const { email, conversation_history = [], subscribe_marketing, session_id, fb = {} } = body;
  if (!email) return errorResponse('email is required', 400, env);
  const anchorAnswer = conversation_history[0]?.answer || '';

  try {
    await upsertKlaviyoProfile(email, {
      'Tara Hair Quiz - Completed': true, 'Tara Hair Quiz - Date': new Date().toISOString(),
      'Tara Hair Quiz - Primary Concern': anchorAnswer, 'Tara Hair Quiz - Total Questions': conversation_history.length,
      'Tara Hair Quiz - Session ID': session_id || '',
    }, env);

    if (env.KLAVIYO_API_KEY) {
      await fetch('https://a.klaviyo.com/api/events/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Klaviyo-API-Key ${env.KLAVIYO_API_KEY}`, 'revision': '2024-10-15' },
        body: JSON.stringify({ data: { type: 'event', attributes: { metric: { data: { type: 'metric', attributes: { name: 'Tara Hair Quiz Completed' } } }, profile: { data: { type: 'profile', attributes: { email } } }, properties: { conversation_history, total_questions: conversation_history.length, primary_concern: anchorAnswer } } } }),
      });
    }
    if (subscribe_marketing && env.KLAVIYO_LIST_ID) await subscribeToKlaviyoList(email, env);
  } catch (e) { console.error('Klaviyo email error:', e); }

  try { await sendFacebookCAPI(email, fb, request, env); } catch (e) { console.error('Facebook CAPI error:', e); }
  return jsonResponse({ success: true }, 200, env);
}

/* ═══════════════════════════════════════════════════════════
   Facebook Conversions API (Server-Side)
   ═══════════════════════════════════════════════════════════ */

async function sendFacebookCAPI(email, fb, request, env) {
  if (!env.FB_PIXEL_ID || !env.FB_CAPI_ACCESS_TOKEN) return;
  const hashSHA256 = async (value) => {
    const encoded = new TextEncoder().encode(value.trim().toLowerCase());
    const hashBuffer = await crypto.subtle.digest('SHA-256', encoded);
    return Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('');
  };
  const userData = { em: [await hashSHA256(email)], client_user_agent: request.headers.get('user-agent') || '', client_ip_address: request.headers.get('cf-connecting-ip') || request.headers.get('x-forwarded-for') || '' };
  if (fb.fbp) userData.fbp = fb.fbp;
  if (fb.fbc) userData.fbc = fb.fbc;
  const eventData = { event_name: 'Lead', event_time: Math.floor(Date.now() / 1000), action_source: 'website', user_data: userData, custom_data: { content_name: 'Tara Hair Consultation' } };
  if (fb.event_id) eventData.event_id = fb.event_id;
  if (fb.event_source_url) eventData.event_source_url = fb.event_source_url;
  const res = await fetch(`https://graph.facebook.com/v21.0/${env.FB_PIXEL_ID}/events`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: [eventData], access_token: env.FB_CAPI_ACCESS_TOKEN }) });
  if (!res.ok) console.error('Facebook CAPI error:', await res.text());
}

/* ═══════════════════════════════════════════════════════════
   Klaviyo Helpers
   ═══════════════════════════════════════════════════════════ */

async function upsertKlaviyoProfile(email, properties, env) {
  if (!env.KLAVIYO_API_KEY) return;
  const res = await fetch('https://a.klaviyo.com/api/profile-import/', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Klaviyo-API-Key ${env.KLAVIYO_API_KEY}`, 'revision': '2024-10-15' }, body: JSON.stringify({ data: { type: 'profile', attributes: { email, properties } } }) });
  if (!res.ok) console.error('Klaviyo upsert error:', await res.text());
}

async function subscribeToKlaviyoList(email, env) {
  if (!env.KLAVIYO_API_KEY || !env.KLAVIYO_LIST_ID) return;
  const res = await fetch(`https://a.klaviyo.com/api/lists/${env.KLAVIYO_LIST_ID}/relationships/profiles/`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Klaviyo-API-Key ${env.KLAVIYO_API_KEY}`, 'revision': '2024-10-15' }, body: JSON.stringify({ data: [{ type: 'profile', id: email }] }) });
  if (!res.ok) console.error('Klaviyo subscribe error:', await res.text());
}

/* ═══════════════════════════════════════════════════════════
   JSON Parsing
   ═══════════════════════════════════════════════════════════ */

function stripCodeFences(raw) { return raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim(); }

function parseQuestionJson(raw) {
  try {
    const result = JSON.parse(stripCodeFences(raw));
    result._parsed = true;
    return result;
  } catch (e) {
    console.error('Failed to parse question AI response:', raw);
    return {
      _parsed: false, _raw: raw,
      question_text: 'How would you describe your scalp right now?', question_context: '',
      options: [
        { label: 'Dry and tight', value: 'dry_tight' }, { label: 'Oily or greasy', value: 'oily' },
        { label: 'Itchy or flaky', value: 'itchy_flaky' }, { label: 'Feels fine, no issues', value: 'normal' },
      ],
      complete: false, micro_insight: '', bridge_text: '',
      diagnostic_memory: { dominant_pattern_emerging: '', covered_domains: [], evidence_map: '{}', biggest_gap: '', user_frustration_theme: '', solution_readiness: 'low' },
    };
  }
}

function parseAssessmentJson(raw) {
  try {
    const result = JSON.parse(stripCodeFences(raw));
    if (!result.concern_level && !result.assessment_headline) {
      console.error('Assessment JSON parsed but missing required fields:', Object.keys(result));
      throw new Error('Incomplete assessment response');
    }
    return result;
  } catch (e) {
    console.error('Failed to parse assessment AI response:', raw);
    return {
      concern_level: 'moderate',
      assessment_headline: 'Your Hair Needs a Scalp-First Approach',
      assessment_summary: 'Based on your answers, your hair concerns trace back to the scalp environment. Tara\'s botanical-powered formulas work from the root to restore balance and vitality.',
      what_is_happening: 'Your scalp\u2019s natural ecosystem \u2014 the balance of oils, moisture, and follicle health that determines how your hair grows and feels \u2014 has been disrupted. This creates a cycle where the hair fiber shows symptoms (dryness, breakage, thinning) that actually originate at the scalp level.',
      key_insight: 'The pattern you described suggests your scalp environment is the primary driver. Surface-level products can\u2019t address this because they don\u2019t reach where the issue begins.',
      why_previous_products_underperformed: 'Most hair care products focus on coating the hair fiber with silicones or temporary smoothing agents. They mask symptoms without addressing the scalp ecosystem where hair health is determined.',
      care_protocol: { entry_point: 'full_routine', intensive_phase: 'Use your recommended Tara routine consistently for the first 4 weeks to reset the scalp environment.', maintenance_phase: 'Continue 2-3 times per week for ongoing scalp and hair health.', expected_timeline: 'Noticeable improvements in scalp comfort and hair vitality within 2-4 weeks.', personalized_note: 'Consistency is key \u2014 the scalp ecosystem needs time to rebalance and support healthier growth.' },
      how_tara_works_for_you: 'Tara combines high-potency botanical extracts with advanced dermatological actives to nourish the scalp ecosystem where hair health begins. Unlike surface-level products, Tara addresses the root cause \u2014 literally.',
      ingredient_highlights: [
        { name: 'Black Garlic Extract', benefit: 'Rich in antioxidants, strengthens hair from root to tip and supports scalp circulation' },
        { name: 'Ceramides', benefit: 'Restore the natural protective barrier of both scalp and hair fiber' },
      ],
      purchase_nudge: 'Your scalp is ready for a reset \u2014 the right botanical actives can restore the balance your hair needs to thrive.',
      klaviyo_properties: { concern_level: 'moderate', primary_concern: 'scalp imbalance', concern_pattern: 'scalp-driven hair concerns' },
    };
  }
}

/* ═══════════════════════════════════════════════════════════
   Fallback Questions — Tara Scalp-First
   ═══════════════════════════════════════════════════════════ */

const FALLBACK_QUESTIONS = [
  { question_text: 'How would you describe your scalp right now?', question_context: '', options: [{ label: 'Dry and tight', value: 'dry_tight' }, { label: 'Oily or greasy', value: 'oily' }, { label: 'Itchy or flaky', value: 'itchy_flaky' }, { label: 'Feels fine, no issues', value: 'normal' }] },
  { question_text: 'How often do you wash your hair?', question_context: 'Washing frequency affects the scalp\'s natural oil balance and follicle health.', options: [{ label: 'Every day', value: 'daily' }, { label: 'Every 2-3 days', value: 'every_2_3' }, { label: 'Once a week', value: 'weekly' }, { label: 'Less than once a week', value: 'rarely' }] },
  { question_text: 'Have you noticed any changes in your hair over the past year?', question_context: 'Changes over time help us understand whether this is a new pattern or a long-standing concern.', options: [{ label: 'More hair fall than before', value: 'increased_fall' }, { label: 'Getting drier or more brittle', value: 'drier_brittle' }, { label: 'Less volume or density', value: 'less_volume' }, { label: 'No significant changes', value: 'stable' }] },
  { question_text: 'Do you use heat styling tools regularly?', question_context: 'Heat exposure can compound scalp and hair concerns over time.', options: [{ label: 'Rarely or never', value: 'rarely' }, { label: 'Once or twice a week', value: '1_2_weekly' }, { label: '3-5 times a week', value: '3_5_weekly' }, { label: 'Almost every day', value: 'daily' }] },
  { question_text: 'What matters most to you in a hair care routine?', question_context: 'This helps us prioritize what matters most to you.', options: [{ label: 'Stopping hair fall', value: 'stop_fall' }, { label: 'A healthy, balanced scalp', value: 'scalp_health' }, { label: 'Softness and shine', value: 'softness_shine' }, { label: 'Clean, natural ingredients', value: 'clean_ingredients' }] },
];

function getFallbackQuestion(turnCount) {
  const idx = Math.min(turnCount, FALLBACK_QUESTIONS.length - 1);
  const q = FALLBACK_QUESTIONS[idx];
  return { ...q, complete: turnCount >= 4, micro_insight: '', bridge_text: '', diagnostic_memory: { dominant_pattern_emerging: '', covered_domains: [], evidence_map: '{}', biggest_gap: '', user_frustration_theme: '', solution_readiness: 'low' } };
}
