# Tara AI Quiz Worker

## Overview

This is a **Cloudflare Worker** that powers Tara's AI-driven Hair & Scalp Consultation quiz. It is a single-file deployment (`tara-quiz-ai-worker.js`) designed to be deployed via the Cloudflare dashboard.

The quiz guides users through an adaptive diagnostic conversation about their hair and scalp concerns, then generates a personalized assessment with product recommendations aligned to Tara's scalp-first philosophy.

## Architecture

**Single file:** `tara-quiz-ai-worker.js` — all logic lives here.

### API Endpoints (all POST)

| Endpoint | Purpose | Default Model |
|---|---|---|
| `/api/question` | Adaptive diagnostic questions — asks one question at a time based on prior answers | gpt-5-nano |
| `/api/assessment` | Generates the full Hair & Scalp Assessment Profile with product recommendations | gpt-5.4 + file_search |
| `/api/email` | Sends quiz results to Klaviyo (profile creation + event tracking) | N/A |

### Flow

1. The Shopify frontend (`tara-quiz.js`) sends the user's answer to `/api/question`
2. The worker uses OpenAI Responses API + vector store search to generate the next best diagnostic question
3. After ~5 questions, the worker signals `complete: true`
4. The frontend calls `/api/assessment` which generates a comprehensive personalized assessment
5. The frontend redirects to the results page where the assessment is rendered
6. Optionally, `/api/email` sends the assessment to Klaviyo for follow-up marketing

### Key Integrations

- **OpenAI Responses API** — primary AI provider for question generation and assessment
- **OpenAI Vector Stores** — hair science / ingredient knowledge base for grounding responses
- **Anthropic Claude** — optional fallback provider
- **Klaviyo** — email marketing (profile + event tracking)
- **Cloudflare KV** (`QUIZ_SESSIONS`) — session state persistence across questions

## Environment Variables

### Secrets
- `OPENAI_API_KEY` — required
- `ANTHROPIC_API_KEY` — optional fallback
- `KLAVIYO_API_KEY` — for email endpoint

### Configuration
- `ALLOWED_ORIGIN` — CORS origin (the Shopify store domain)
- `DEFAULT_PROVIDER` — `"openai"` or `"anthropic"`
- `KLAVIYO_LIST_ID` — Klaviyo list for quiz subscribers
- `OPENAI_VECTOR_STORE_IDS` — comma-separated, for assessment file_search
- `OPENAI_QUESTION_VECTOR_STORE_IDS` — comma-separated, for question direct search

### KV Namespace
- `QUIZ_SESSIONS` — must be bound in Cloudflare dashboard

## Development Notes

- This is deployed as a single file via the Cloudflare Workers dashboard — there is no build step or bundler
- The companion frontend lives in `narhamah/tara-saudi-shopify` (Shopify theme)
- System prompts are defined inline as constants (`DIAGNOSTIC_SYSTEM_PROMPT`, `ASSESSMENT_SYSTEM_PROMPT`)
- The quiz uses 7 assessment domains: scalp_health, hair_fall, damage_structure, hydration_moisture, texture_manageability, routine_habits, environment_lifestyle
- Questions complete after 5 answered questions (`complete: true`)
- Assessment returns JSON with: concern_level, assessment_headline, assessment_summary, what_is_happening, key_insight, recommended_products, care_protocol

## Brand Context

Tara is a scalp-first hair care brand. Key principles:
- Scalp health is the foundation of hair health
- Clean beauty: sulfate-free, silicone-free, paraben-free, cruelty-free
- Manufactured in Spain under EU cosmetic regulations
- Key ingredients: Black Garlic Extract, Ceramides, Niacinamide, Biotin, Salicylic Acid, Zinc Pyrithione
- Tone: warm, knowledgeable, empowering — like a trusted hair health specialist
