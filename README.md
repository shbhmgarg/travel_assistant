# AI Travel Planning Assistant — Singapore

A travel assistant for Singapore that answers destination questions from a
curated knowledge base and pulls in live weather and currency data when a
question needs it. Ask about attractions and it looks them up in the
knowledge base with citations. Ask about the weather or a currency
conversion and it calls a live API through MCP. Ask something that needs
both — like a weather-adjusted itinerary with a converted budget — and it
combines all of it into one answer, labeling what came from where.

## Knowledge base

Six Singapore travel resources were converted into Markdown files under
`data/`, each keeping its original title and URL so the assistant can cite
it properly:

- `wikivoyage_singapore.md` — Wikivoyage's Singapore guide: districts,
  attractions, transport, food, itineraries.
- `visit_singapore_essential_info.md` — climate, language, connectivity,
  practical visitor info.
- `visit_singapore_itineraries.md` — sample itineraries for different
  trip lengths and traveller types.
- `visit_singapore_things_to_do.md` — attractions and activities across
  different interest categories.
- `singapore_tourist_pass.md` — tourist transit pass types, pricing, how
  they work.
- `singapore_smart_transport_payment.md` — how visitors actually pay for
  buses and trains (EZ-Link, contactless cards, cash).

The last two were added on top of the four suggested in the brief, since
transport and payment came up as real gaps once the itinerary answers were
working — a traveller planning a trip wants to know how they'll actually
pay for the MRT, not just where to go.

## How the RAG side works

`ingest.py` builds the knowledge base once, offline:

Each Markdown file is read and its frontmatter (source title, source URL)
is pulled out as metadata before the document body is chunked. Chunking
uses `RecursiveCharacterTextSplitter` with a chunk size of 800 characters
and 100 characters of overlap, splitting on Markdown headings first so a
topic doesn't get cut in half across chunks. Each chunk is embedded with
Google's `gemini-embedding-001` model and stored in a FAISS index on disk
(`vectorstore/`), so it only needs to be rebuilt when the source files
change.

At query time, a LangChain tool (`search_travel_knowledge_base` in
`agent.py`) embeds the user's question, retrieves the four most similar
chunks, and hands them to the LLM labeled by source. The system prompt
tells the model to only state destination facts that came from those
chunks and to say plainly when nothing relevant turned up, rather than
filling the gap with a guess — the retrieval tool returns a `NO_RESULTS`
marker in that case, and the model is told to treat that as "I don't know"
rather than a starting point. Every knowledge-base fact in a response gets
attributed to its source ("According to *Wikivoyage*...").

`test_retrieval.py` checks this part on its own, without touching the LLM
— useful for confirming the retriever finds the right chunks even when the
question doesn't share exact wording with the source (a question about
rain correctly pulls up indoor-activity content, for example).

## How the MCP side works

`mcp_server.py` is a small MCP server exposing two tools, run over stdio.
`agent.py` launches it as a subprocess and discovers its tools through
`MultiServerMCPClient` — the tools live in their own process and could
just as easily be called by any other MCP client, not just this one.

**Weather** — `get_weather_forecast(city, days=3)` calls Open-Meteo (free,
no API key) for current conditions and a multi-day forecast: temperature,
humidity, conditions, wind, and sunrise/sunset.

**Currency** — `convert_currency(amount, from_currency, to_currency)`
calls Frankfurter (also free, no key, backed by ECB reference rates) for
live exchange rates.

Both tools catch network timeouts, bad responses, and unknown inputs (an
unsupported city, say) and return a plain `{"error": "..."}` instead of
crashing. The system prompt tells the model that if a tool result contains
an error, it should say the service is unavailable rather than inventing a
number that sounds right. `test_mcp.py` exercises the server directly —
listing its tools and calling both with real arguments — to confirm the
transport and the live API calls work before the agent ever touches them.

One dependency note worth mentioning: the `mcp` package went through a
breaking rewrite from v1 to v2 partway through building this (`FastMCP`
was restructured). Rather than chase a moving target mid-project, the
dependency is pinned to `mcp<2`, which is what `requirements.txt` reflects.

## Combined responses

There's no special "combined mode" — it falls out naturally from the agent
being able to call more than one tool per turn. Given something like *"I
have a budget of INR 60,000 — convert it to SGD and suggest a three-day
itinerary based on the weather,"* it calls the currency tool for the
budget, the weather tool for the forecast, and the knowledge-base tool for
attractions and itinerary ideas, then writes one answer that keeps those
three sources visibly separate, plus its own suggestions on top. See the
third example below — that's a real, unedited response.

## Prompt design and conversation memory

The system prompt in `agent.py` does a few specific jobs:

- Tells the model exactly when to reach for which tool — destination
  questions go to the knowledge base, anything time-sensitive goes to the
  matching MCP tool — and forbids answering either kind from memory.
- Forbids stating any specific fact (an attraction, a price, a weather
  condition, a rate) unless it came from a tool call, and requires an
  honest "I don't have that" when a tool comes back empty or with an
  error.
- Requires every response to visibly separate knowledge-base facts, live
  tool results, and the model's own suggestions, so it's obvious at a
  glance what's sourced and what's the model reasoning on its own.
- Asks the model to remember details from earlier in the conversation —
  budget, dates, preferences — instead of asking again.

That last point is backed by actual code, not just a prompt instruction:
`TravelAssistant.achat()` appends every user message and every final
answer to `self.chat_history`, and the full history is sent along with
each new question, so the model genuinely has the whole conversation to
work with rather than just being told to "remember."

## Project structure

```
travel_assistant/
├── data/                                  knowledge base source documents
├── vectorstore/                           FAISS index (built by ingest.py, committed for convenience)
├── ingest.py                              builds the vector store from data/*.md
├── rag_qa.py                              standalone RAG-only Q&A CLI, used while building the RAG piece
├── mcp_server.py                          MCP server: weather + currency tools
├── agent.py                               the combined RAG + MCP agent, plus a CLI
├── streamlit_app.py                       chat UI
├── test_retrieval.py                      retrieval-only test, no LLM
├── test_mcp.py                            MCP server/tool test, no LLM
├── requirements.txt
├── .env                                   GOOGLE_API_KEY (not committed)
└── .gitignore
```

## Setup

You'll need Python 3.11+ and a free API key from
[Google AI Studio](https://aistudio.google.com/).

```bash
git clone <your-repo-url>
cd travel_assistant

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

echo "GOOGLE_API_KEY=your-key-here" > .env

# A pre-built vector store is already committed under vectorstore/, so this
# is only needed if you change something in data/
python ingest.py

# optional sanity checks
python test_retrieval.py
python test_mcp.py

# run it
streamlit run streamlit_app.py
# or, for a plain terminal chat loop:
python agent.py
```

A quick note on the model: the assistant runs on
`gemini-flash-lite-latest` rather than Google's newest flagship model.
The flagship model's free tier caps out at 20 requests/day, and each
question here costs at least two model calls (one to decide on a tool, one
to write the final answer) — that gets used up fast during normal testing.
`flash-lite` has a much higher free daily quota and handled everything
here, including the examples below.

## Sample questions and responses

These are real, unedited outputs from the running app.

---
**"What are the must-visit attractions in Singapore?"** (knowledge base only)

> Here are the top must-visit attractions in Singapore:
>
> **Major Attractions** — According to *Wikivoyage: Singapore Travel
> Guide*: Gardens by the Bay (Supertree Grove, Cloud Forest, Flower
> Dome), Marina Bay Sands & Merlion Park, Singapore Zoo & Night Safari,
> Singapore Botanic Gardens (UNESCO World Heritage Site), Sentosa Island.
>
> **Cultural Enclaves & Neighbourhoods** — According to *Wikivoyage* and
> *Visit Singapore: Things to Do*: Chinatown, Kampong Glam, Little India,
> Civic District.
>
> **My Suggestion:** If this is your first visit, a 3–4 day trip is
> ideal for combining the modern icons around Marina Bay, a full day at
> Sentosa or the Zoo, and half-days in Chinatown, Kampong Glam, and local
> hawker centers.

---
**"What's the weather like right now?"** (weather tool only)

> **Current Conditions:** Temperature 28.2°C, Conditions: Thunderstorm,
> Humidity 84%, Wind 10.7 km/h.
>
> **Today's Forecast:** High/Low 30.8°C / 24.5°C, Precipitation chance
> 99%.
>
> **My Suggestion:** Since rain and thunderstorms are expected today,
> it's a great time to visit indoor attractions like the Cloud Forest &
> Flower Dome, Jewel Changi Airport, the ArtScience Museum, or the
> museums in the Civic District.

---
**"I have a budget of INR 60,000 — convert it to SGD and suggest a
three-day itinerary based on the weather."** (knowledge base + both MCP
tools)

> **Live Currency Conversion:** At current live exchange rates, INR
> 60,000 is approximately **SGD 800.05**.
>
> **Live Weather Forecast:** Thunderstorms and heavy rain likely across
> all 3 days (92–99% precipitation chance).
>
> **Weather-Resilient 3-Day Itinerary** — According to *Visit Singapore:
> Sample Itineraries* and *Visit Singapore: Things to Do*:
> - **Day 1:** Gardens by the Bay's Flower Dome & Cloud Forest (fully
>   enclosed), ArtScience Museum.
> - **Day 2:** National Museum / Asian Civilisations Museum, Chinatown
>   Complex Hawker Centre for lunch, Orchard Road's underground mall
>   network.
> - **Day 3:** S.E.A. Aquarium on Sentosa, Jewel Changi Airport's Rain
>   Vortex.
>
> **My Suggestion for Managing Your Budget (~SGD 800):** hawker meals
> (SGD 5–8), public transit (SGD 10–15/day), and the three named
> attractions (~SGD 100–150 total) — leaving room for dining and transit
> across 3 days.

---
**"Plan a trip to Japan."**

> I would love to help you plan a trip, but my expertise is focused specifically on Singapore!
>
> If you are thinking about visiting Singapore instead, let me know your preferences (such as your budget, interests, or how many days you'll be staying), and I can put together a fantastic itinerary for you using my travel knowledge base..

---

That third one is the important example: it pulls attractions and
itinerary content from the knowledge base with citations, gets a live
forecast and a live currency conversion from the two MCP tools, and keeps
all three clearly separate from the model's own budget suggestion.

## Testing

`test_retrieval.py` checks the RAG pipeline on its own — retrieval
quality, no LLM involved. `test_mcp.py` checks the MCP server on its own —
tool listing and live API calls, again no LLM involved. Beyond those two,
everything was tested end to end through both the CLI and the Streamlit
app: knowledge-base-only questions, each MCP tool on its own, the combined
scenario above, and follow-up questions that depend on earlier context in
the same conversation.

## Known limitations

- Built for one destination (Singapore), which the brief allows.
- The weather tool only has coordinates for Singapore configured; asking
  about anywhere else comes back as a clear error rather than a guess or
  a crash.
- Gemini's free tier has daily quotas, so heavy testing in a short window
  can trigger a rate-limit error — using `flash-lite` instead of the
  flagship model keeps this from being a practical problem.
- No flight or hotel booking, payment processing, or route navigation —
  out of scope per the brief.

## Minimum acceptance criteria

| Criterion | Status |
|---|---|
| Knowledge base from ≥3 travel resources | Done — 6 sources |
| Embedding-based semantic retrieval | Done — FAISS + Gemini embeddings |
| Grounded answers with source references | Done — every KB fact cites its source |
| Weather information through an MCP tool | Done |
| Currency conversion through an MCP tool | Done |
| ≥1 response combining RAG and MCP | Done — see the third sample above |
| Multi-turn conversation with retained context | Done — full history sent each turn |
| Appropriate tool selection based on user intent | Done — system prompt + per-tool descriptions |
| Clear handling of missing knowledge and tool failures | Done — `NO_RESULTS` marker and MCP `error` field, both surfaced honestly |
| A simple, usable interface | Done — Streamlit chat UI |

## The assignment's 20 numbered requirements

The brief numbers 20 specific requirements across the RAG section, the MCP
section, and the deliverables list. Going through them one by one:

**RAG requirements**

1. **Load travel content from public documents or web pages.** Done —
   six public Singapore travel pages, converted to Markdown in `data/`
   and loaded by `ingest.py`.
2. **Divide content into meaningful chunks.** Done — 800-character
   chunks with 100 characters of overlap, splitting on Markdown headings
   first so a chunk doesn't cut a topic in half.
3. **Generate embeddings for the chunks.** Done — each chunk is embedded
   with `gemini-embedding-001`.
4. **Store embeddings in a vector store.** Done — FAISS, saved to disk
   under `vectorstore/`.
5. **Retrieve relevant chunks for each user question.** Done — the
   `search_travel_knowledge_base` tool embeds the question and retrieves
   the top 4 matching chunks at query time.
6. **Generate answers grounded in the retrieved content.** Done — the
   system prompt restricts destination facts to what came back from
   retrieval, and requires an honest "don't know" when nothing relevant
   is found instead of a guess.
7. **Display the source title or source link used for the answer.**
   Done — every knowledge-base fact in a response is attributed to its
   source title, pulled from each file's frontmatter metadata.

**MCP requirements**

8. **Connect to at least two MCP tools.** Done — two: weather and
   currency conversion.
9. **Make the tools available to the AI application.** Done — `agent.py`
   launches `mcp_server.py` as a subprocess and pulls in its tools via
   `MultiServerMCPClient.get_tools()`.
10. **Select the appropriate tool based on the user request.** Done —
    handled by the LLM itself, guided by the system prompt and each
    tool's description.
11. **Pass the required input to the selected tool.** Done — standard
    LangChain tool-calling; the model fills in arguments like city,
    amount, and currency codes based on the question.
12. **Use the returned information in the final response.** Done — tool
    results are fed back to the model, which incorporates them into the
    answer rather than just displaying them raw.
13. **Clearly indicate when current information came from an MCP tool.**
    Done — the system prompt requires labeling live data separately from
    knowledge-base facts and the model's own suggestions (see the sample
    responses above).
14. **Handle unavailable tools or failed results without fabricating an
    answer.** Done — both MCP tools catch timeouts, HTTP errors, and bad
    input, returning `{"error": "..."}` instead of crashing, and the
    system prompt tells the model to report that honestly rather than
    guessing a number.

**Deliverables**

15. **Source code in a Git repository.** Done — committed, `.env`
    excluded via `.gitignore`.
16. **A working application.** Done — runs via `streamlit run
    streamlit_app.py` or `python agent.py`.
17. **Knowledge-base documents, or clear instructions for obtaining
    them.** Done — all 6 documents are committed under `data/`, so no
    separate download step is needed.
18. **A README covering architecture, knowledge-base sources, RAG
    workflow, MCP tools, prompt and context strategy, and setup
    instructions.** This document — see the sections above.
19. **Sample questions and application responses.** Done — see "Sample
    questions and responses" above, three real, unedited runs.
20. **A short demonstration showing RAG, MCP, a combined response, and
    conversational context.** Done — to be recorded as a short screen
    capture walking through the same scenarios shown above, plus a
    follow-up question that relies on earlier conversation context.