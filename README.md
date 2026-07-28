# CarWise Cambodia

CarWise is my CS382 final project. It is a RAG search system for finding cars
listed in Cambodia. A user can describe the type of car they want, set filters,
and view matching listings as cards. The generated answer is based on retrieved
listing data and includes source citations.

## Main features

- Natural-language car search
- Filters for price, make, year, condition, location, and body type
- Cards with photos, prices, fuel economy, engine size, seats, and transmission. Informations that Khmer24 itself doesn't provide sometime.
- Similar-price recommendations
- Expandable source chunks with similarity scores
- Local LLM, OpenAI, Ollama, and deterministic fallback options
- Clear failure messages when the dataset cannot support a question

## Dataset

The project uses a saved snapshot of public
[Khmer24 car listings](https://www.khmer24.com/c-cars-for-sale) collected in
July 2026. Saving the data locally makes the demo reproducible even when a
listing is changed or removed.

The dataset contains:

- 600 unique listings
- 51 identified makes
- 281 model labels
- 475 gasoline vehicles
- 69 electric vehicles
- 14 plug-in hybrids
- 3,000 searchable chunks

Vehicle specifications were filled from exact model-year references when
possible. Some missing fields were matched with FuelEconomy.gov or official
manufacturer information. These specifications may still differ from the exact
trim imported into Cambodia.

## How it works

```text
Khmer24 snapshot
      |
      v
documents -> section-aware chunks -> MiniLM embeddings -> cosine search
                                                        |
user question -> filters -------------------------------+
                                                        |
                                                        v
                                      LLM -> cited answer
```

Each vehicle is divided into one complete profile and smaller section chunks.
The app uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and cosine
similarity for ranking. Price, make, year, fuel type, and body type are also
used as strict filters.

I chose MiniLM because it runs locally and is fast enough for a live demo. A
simple NumPy index is enough for 3,000 chunks, and the section-aware chunking
keeps specifications together while still providing readable source excerpts.

All results above the similarity threshold are displayed as cards. The selected
top-k results are passed to the answer generator. The code is separated into:

- `carwise/cambodia_dataset.py` - data loading and validation
- `carwise/chunking.py` - document chunking
- `carwise/embeddings.py` - embedding model
- `carwise/vector_store.py` - filtering and similarity search
- `carwise/generation.py` - answer generation
- `carwise/pipeline.py` - RAG pipeline
- `app.py` - Streamlit interface

## Setup

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/pteang/CS-382-Final-Project-CarWise.git
cd CS-382-Final-Project-CarWise

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/prepare_dataset.py
python scripts/build_index.py
python -m streamlit run app.py
```

The first installation and model download can take several minutes because
PyTorch and the local models are fairly large. Do not interrupt the installation.

The default Local LLM does not require an API key. OpenAI can be enabled with:

```bash
export OPENAI_API_KEY="your-key"
python -m streamlit run app.py
```

## Tests and evaluation

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The full nine-query evaluation is in
[`evaluation/report.md`](evaluation/report.md). The recorded results were:

- Retrieval evidence coverage: 9/9
- Valid citation grounding: 9/9
- Manual answer-quality pass: 4/9
- Mean retrieval time: 0.01 seconds
- Mean local generation time: 4.79 seconds

Retrieval worked well, but the small local model sometimes used awkward wording
or made a comparison that was not fully supported. The listing cards and source
panel are therefore kept visible so the user can check the evidence.

## Known limitations

- Prices, conditions, and availability can change after the snapshot date.
- Technical specifications may not match the exact Cambodian import trim.
- The dataset does not verify crash safety, reliability, repair cost, accident
  history, legal ownership, or fair market value.
- The local LLM is convenient for an offline demo but is less accurate than a
  larger hosted model.
- The first run is slower because the embedding and generation models must load.

Questions about unsupported topics such as crash safety or reliability are
rejected instead of producing an unsupported recommendation.

---

Phalborith Teang - CS 382 Final Project - Paragon International University
