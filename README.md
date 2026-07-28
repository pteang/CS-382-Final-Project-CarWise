# CarWise Cambodia

CarWise is a Retrieval-Augmented Generation (RAG) search system for grounded car
recommendations in Cambodia. A shopper describes a vehicle need, CarWise applies
strict constraints such as budget and model year, retrieves matching marketplace
listing documents, and produces a cited answer with expandable evidence.

Results appear as visual vehicle cards with the marketplace photo, asking price,
fuel economy, engine, seating, and transmission. A second card row suggests other
listings with the closest asking prices.

## Dataset

The default corpus is a timestamped snapshot of public vehicle listings from
[Khmer24 Cars for Sale](https://www.khmer24.com/c-cars-for-sale), observed on
2026-07-26 and expanded with diverse listings observed on 2026-07-28. It
contains 600 unique listings across 51 identified makes (52 make labels when
the `Not identified` bucket is counted) and 281 model labels, including 475
gasoline vehicles, 69 electric vehicles, 14 plug-in hybrids, and 76 sports,
coupe, or convertible listings. This is well above the 20 documents required
by the project brief.

Each listing document includes:

- Seller asking price in USD
- Make, model, and model year when identifiable
- Seller condition and paperwork labels
- Cambodian location and province/municipality
- Normalized body type and fuel/powertrain when supported by the title
- Reference fuel economy, cylinders, displacement, seating, and transmission
- Marketplace listing photo URL
- Observation date and a direct link to the source listing

Seller phone numbers are not stored. The checked-in snapshot makes the classroom demo
reproducible even if listings change or disappear.

## Evidence boundaries

CarWise can recommend from the snapshot by asking price, make, model year, body type,
fuel type, condition label, and Cambodian location. Price and availability are
time-sensitive seller claims.

Technical specifications are stored separately in
`data/raw/reference_vehicle_specs.csv`. Exact model-year references are used when
available. When a listing does not identify its engine, transmission, drive system, or
trim, the interface shows a manufacturer model-family range when one is supportable.
If the seller does not identify enough of the vehicle to support a responsible lookup,
the card explicitly says which configuration detail is missing instead of leaving the
field blank or inventing a precise number.

For 211 listings, missing fuel economy, cylinder count, displacement, and
transmission fields were matched conservatively against the official
[FuelEconomy.gov data download](https://www.fueleconomy.gov/feg/ws/index.shtml)
using exact model year and make plus a specific model-family match. EPA values
describe U.S.-market configurations, so the card displays a range when multiple
configurations exist and warns that a Cambodian import may use a different trim.
Another 77 listings use official manufacturer model-family references, including
Toyota, Kia, Hyundai, Ford, Mazda, Mitsubishi, Suzuki, McLaren, BYD, Honda, Peugeot,
XPENG, and ZEEKR sources. Seating is numeric or a manufacturer-supported range when
available; otherwise the field clearly describes the unresolved trim or body
configuration.

The dataset does **not** verify:

- Crash safety
- Reliability or maintenance cost
- Mechanical condition or accident history
- Legal ownership or paperwork authenticity
- Fair market value

Queries that require crash safety, reliability, maintenance cost, or comfort are
rejected before retrieval. The interface tells the user what the corpus can support
instead of returning an unrelated vehicle.

## Architecture

```text
Khmer24 snapshot CSV
        |
        v
listing documents -> section-aware chunks -> sentence embeddings
                                                |
User query -> budget/year/type filters -> cosine vector retrieval
                                                |
                                                v
                         grounded summary or LLM -> cited recommendation
```

The implementation is separated into:

- `carwise/cambodia_dataset.py`: listing validation and document ingestion
- `carwise/chunking.py`: section-aware chunking
- `carwise/embeddings.py`: local sentence-transformer embeddings
- `carwise/vector_store.py`: strict constraints and persistent cosine index
- `carwise/generation.py`: local, OpenAI, and Ollama grounded generation
- `carwise/pipeline.py`: unsupported-requirement checks and orchestration
- `app.py`: Streamlit interface

The older `carwise/dataset.py` remains only as an optional reference implementation
for EPA ingestion; it is no longer the default corpus.

## Design decisions and measured outcomes

### Dataset and ingestion

A checked-in marketplace snapshot was chosen instead of scraping during app startup.
This keeps the classroom demonstration reproducible and avoids changing results when a
seller edits or removes a listing. The 600 listing documents are large enough to test
retrieval diversity while remaining small enough for a local index.

### Chunking

Each vehicle produces one complete-profile chunk plus section-specific chunks. Long
sections are split near sentence boundaries at 1,100 characters with 160 characters of
overlap. The overview chunk keeps price, vehicle class, fuel type, and specifications
together for multi-constraint questions; the smaller section chunks retain precise
citation context. The current corpus contains 3,000 searchable chunks.

### Embeddings and vector retrieval

`sentence-transformers/all-MiniLM-L6-v2` was selected because it provides real semantic
embeddings locally without an API key and has low enough latency for a live demo.
Normalized vectors are stored in a persistent NumPy matrix and ranked with cosine
similarity. A hosted vector database would add operational complexity without improving
this 3,000-chunk classroom corpus.

Semantic ranking is combined with strict metadata constraints inferred from the query,
such as maximum price, model year, make, fuel type, and body type. This prevents a
semantically similar but over-budget vehicle from being recommended. Vector retrieval
ranks every distinct listing that passes those constraints. The answer-evidence control
defaults to `top_k=5`, so generation stays focused while the interface displays every
result at or above the default `0.22` similarity threshold.

### Generation and grounding

The default offline generator is the instruction-tuned
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
model. It receives compact evidence for the strongest profiles selected by the
answer-evidence control, while the interface displays all qualifying vehicle cards and
source records. Compact context reduced mean generation latency to 4.79 seconds in the
recorded nine-query evaluation. A citation guard requires at least one valid `[S#]`
reference before an answer is shown.

The small local model is a deliberate offline-demo trade-off, not the highest-quality
generator. It produced valid citation labels on 9/9 evaluation queries, but manual
answer-quality review passed only 4/9 because several answers used awkward wording or
made unsupported comparisons. OpenAI or Ollama remains available when higher language
quality is required. The visual result cards and expandable source text remain the
authoritative evidence.

### Latency and caching

The embedding model, local generation model, and vector index are cached with
Streamlit's resource cache. The index includes a corpus fingerprint so a data update
automatically invalidates stale vectors. In the latest evaluation, mean retrieval
latency was 0.01 seconds and mean local-LLM generation latency was 4.79 seconds after
models were loaded.

The reproducible expansion inputs and normalization script are:

- `data/raw/khmer24_diverse_catalog_2026-07-28.json`: 1,216 structured marketplace
  results collected across 50 category and make searches
- `scripts/expand_catalog.py`: round-robin selection, conservative normalization,
  photo extraction, and unverified-spec handling
- `scripts/enrich_specs.py`: conservative exact-year/make/model enrichment from
  FuelEconomy.gov, with seller-advertised EV range fallback
- `scripts/complete_specs.py`: official manufacturer-family enrichment and explicit
  nonblank limitations when a seller does not identify the exact configuration

## Quick start

Use the project virtual environment so all commands use the same interpreter:

```bash
cd "/Users/rith_45/Documents/Search Engine"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/build_index.py
python -m streamlit run app.py
```

Do not interrupt the dependency installation. `torch`, which is required by the
embedding model, is a large download. The earlier `numpy` and `streamlit` errors occur
when installation is cancelled before it finishes.

The first index build may download
`sentence-transformers/all-MiniLM-L6-v2`. The first Local LLM search may also download
`Qwen/Qwen2.5-0.5B-Instruct`. Later runs reuse both local model caches.

## Generation options

The default **Local LLM** works without an API key and produces a grounded answer from
the retrieved evidence. The **Deterministic fallback** remains available only for
troubleshooting. OpenAI or Ollama can produce higher-quality conversational comparisons
from the same evidence.

For OpenAI:

```bash
export OPENAI_API_KEY="your-key"
python -m streamlit run app.py
```

For Ollama:

```bash
ollama pull llama3.2:3b
export OLLAMA_MODEL="llama3.2:3b"
python -m streamlit run app.py
```

## Rebuild from the included snapshot

```bash
python scripts/expand_catalog.py
python scripts/enrich_specs.py
python scripts/complete_specs.py
python scripts/prepare_dataset.py
python scripts/build_index.py
```

To use another snapshot with the same schema:

```bash
python scripts/prepare_dataset.py --source-csv /path/to/cambodia_listings.csv
python scripts/build_index.py
```

Required columns are:

```text
listing_id,title,make,model,model_year,price_usd,condition,registration,
location,province,body_type,fuel_type,observed_at,source_url
```

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python scripts/evaluate.py --top-k 5
```

The retrieval evaluation includes budget, make, body type, location, condition,
hybrid, electric, and sports-car queries. Unit tests cover unsupported safety and
reliability handling.

`evaluation/report.md` contains the required per-query retrieval and generation
write-up. The latest recorded run achieved:

- Retrieval evidence coverage: 9/9
- Valid citation grounding: 9/9
- Manual generated-answer quality: 4/9
- Mean retrieval latency: 0.01 seconds
- Mean local-LLM generation latency: 4.79 seconds

The gap between citation validity and answer quality is important: a syntactically valid
citation does not guarantee that a small model paraphrased the evidence correctly.

## Dataset-error recovery

The app checks the document metadata at startup. If older EPA files or a mixed corpus
are found in `data/documents`, it automatically rebuilds the Cambodian documents from
the included snapshot and invalidates Streamlit's cached search resources.

After pulling or copying a newer version of the project, stop any old Streamlit
process with `Ctrl+C` and start it again:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## Suggested live demo

1. Ask: “Which used SUVs in Phnom Penh are listed below $20,000?”
2. Expand the retrieved sources and show the direct Khmer24 links.
3. Ask: “Recommend a Toyota Prius under $18,000.”
4. Change the maximum-price or make filter and repeat.
5. Ask: “What is the safest car under $20,000?” and show that CarWise refuses the
   unsupported safety claim while explaining that price itself is supported.
