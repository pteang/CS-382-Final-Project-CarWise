from __future__ import annotations

import hashlib
import os
import time

import requests
import streamlit as st

from carwise.cambodia_dataset import ensure_cambodia_corpus
from carwise.chunking import chunk_documents
from carwise.config import (
    CAMBODIA_SNAPSHOT_CSV,
    CURATED_CSV,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DOCUMENTS_DIR,
    INDEX_DIR,
    LISTING_IMAGES_CSV,
    REFERENCE_SPECS_CSV,
)
from carwise.corpus import load_documents
from carwise.embeddings import SentenceTransformerEmbedder
from carwise.generation import (
    GenerationError,
    OllamaGenerator,
    OpenAIResponsesGenerator,
    RetrievalPreviewGenerator,
    TransformersGenerator,
)
from carwise.models import Document
from carwise.pipeline import RAGPipeline
from carwise.recommendations import similar_price_documents
from carwise.uploads import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    uploaded_text_documents,
)
from carwise.vector_store import VectorIndex, inferred_price_range


st.set_page_config(
    page_title="CarWise | Grounded Car Recommendations",
    page_icon="🚗",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .block-container { max-width: 1180px; padding-top: 2rem; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #0b7285 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.2rem; }
    .hero p { margin: 0; opacity: .92; }
    .scope-note {
        border-left: 4px solid #f59f00;
        background: #fff9db;
        color: #5f3d00;
        padding: .8rem 1rem;
        border-radius: 8px;
        margin: .5rem 0 1rem 0;
    }
    .metric-row {
        color: #486581;
        font-size: .9rem;
        margin-bottom: .8rem;
    }
    div[data-testid="stImage"] {
        width: 100%;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: 210px;
        object-fit: cover;
        border-radius: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the local embedding model and vector index...")
def load_search_resources(model_name: str, corpus_version: str):
    # corpus_version is intentionally part of the cache key. It prevents Streamlit
    # from reusing old EPA documents after the Cambodian dataset is rebuilt.
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise FileNotFoundError(
            "The vehicle corpus is missing. Run `python scripts/prepare_dataset.py`."
        )
    chunks = chunk_documents(documents)
    embedder = SentenceTransformerEmbedder(model_name)
    if (INDEX_DIR / "index.json").exists() and (INDEX_DIR / "embeddings.npy").exists():
        index = VectorIndex.load(INDEX_DIR)
        if not index.is_current(chunks, model_name):
            index = VectorIndex.build(chunks, embedder)
            index.save(INDEX_DIR)
    else:
        index = VectorIndex.build(chunks, embedder)
        index.save(INDEX_DIR)
    return documents, embedder, index


def document_corpus_version(documents) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.document_id):
        digest.update(document.document_id.encode("utf-8"))
        digest.update(document.text.encode("utf-8"))
    return digest.hexdigest()


@st.cache_resource(show_spinner="Loading the local generation model...")
def load_local_llm(model_name: str):
    return TransformersGenerator(model_name)


def make_generator(provider: str):
    if provider == "Local LLM":
        return load_local_llm(DEFAULT_LOCAL_LLM_MODEL)
    if provider == "OpenAI":
        return OpenAIResponsesGenerator(DEFAULT_OPENAI_MODEL)
    if provider == "Ollama":
        return OllamaGenerator(DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE_URL)
    return RetrievalPreviewGenerator()


def source_documents(
    sources,
    all_documents,
    limit: int | None = None,
) -> list[Document]:
    """Resolve retrieved chunks to unique listing documents in ranked order."""
    documents_by_id = {
        document.document_id: document for document in all_documents
    }
    matches: list[Document] = []
    seen: set[str] = set()
    for source in sources:
        document_id = source.chunk.document_id
        if document_id in seen or document_id not in documents_by_id:
            continue
        seen.add(document_id)
        matches.append(documents_by_id[document_id])
        if limit is not None and len(matches) == limit:
            break
    return matches


def display_value(value, fallback: str = "Not verified") -> str:
    if value is None or str(value).strip() in {"", "None", "nan"}:
        return fallback
    return str(value).strip()


def render_vehicle_cards(
    vehicles: list[Document],
    cards_per_row: int = 3,
) -> None:
    """Render every vehicle in a readable, wrapping card grid."""
    for row_start in range(0, len(vehicles), cards_per_row):
        row_vehicles = vehicles[row_start : row_start + cards_per_row]
        columns = st.columns(cards_per_row, gap="medium")
        for column, vehicle in zip(columns, row_vehicles):
            metadata = vehicle.metadata
            price = metadata.get("price_usd")
            price_label = (
                f"${float(price):,.0f}" if isinstance(price, (int, float))
                else "Price not reported"
            )
            with column:
                with st.container(border=True):
                    image_url = display_value(metadata.get("image_url"), "")
                    if image_url:
                        st.image(
                            image_url,
                            caption="Marketplace listing photo",
                            width="stretch",
                        )
                    else:
                        st.info("🚗 Listing photo unavailable")
                    st.markdown(f"#### {vehicle.title}")
                    st.markdown(f"### {price_label}")
                    st.caption(
                        f"{display_value(metadata.get('condition'), 'Condition unknown')} · "
                        f"{display_value(metadata.get('body_type'), 'Body type unknown')} · "
                        f"{display_value(metadata.get('location'), 'Location unknown')}"
                    )
                    st.markdown(
                        f"**{'Range / efficiency' if metadata.get('fuel_type') == 'Electricity' else 'Fuel economy'}:** "
                        f"{display_value(metadata.get('fuel_economy'))}  \n"
                        f"**Engine:** {display_value(metadata.get('cylinders'))} cylinders · "
                        f"{display_value(metadata.get('displacement_l'))}  \n"
                        f"**Seats:** {display_value(metadata.get('seats'))}  \n"
                        f"**Transmission:** {display_value(metadata.get('transmission'))}"
                    )
                    st.link_button(
                        "View marketplace listing",
                        vehicle.source_url,
                        use_container_width=True,
                    )


st.markdown(
    """
    <div class="hero">
      <h1>CarWise</h1>
      <p>Grounded car recommendations from Cambodian marketplace listings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="scope-note">
      <strong>Evidence scope:</strong> CarWise can compare seller asking price,
      model year, make, body type, fuel type, condition label, and listing location
      in Cambodia. Prices and availability can change. Listings do not prove crash
      safety, reliability, mechanical condition, ownership, or fair market value.
      Technical specifications are model-year references and should be checked
      against the exact imported trim. Uploaded-corpus mode answers only from
      the text files selected for that session.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider(
        "Sources used in written answer",
        2,
        10,
        5,
        help=(
            "This controls the evidence sent to the answer generator. "
            "All qualifying vehicle cards are still displayed."
        ),
    )
    minimum_similarity = st.slider(
        "Minimum similarity", 0.0, 0.8, 0.22, 0.01,
        help="Higher values reject weak matches more aggressively.",
    )
    answer_mode = st.selectbox("Answer mode", ["Concise", "Detailed", "Comparison"])
    default_provider = (
        "OpenAI" if os.getenv("OPENAI_API_KEY") else "Local LLM"
    )
    provider = st.radio(
        "Generation provider",
        ["Local LLM", "OpenAI", "Ollama", "Deterministic fallback"],
        index=[
            "Local LLM",
            "OpenAI",
            "Ollama",
            "Deterministic fallback",
        ].index(default_provider),
    )
    st.caption(
        "Local LLM is the offline graded-demo option. OpenAI and Ollama are "
        "alternatives; deterministic fallback is for troubleshooting only."
    )
    st.divider()
    st.subheader("Upload new corpus")
    uploaded_files = st.file_uploader(
        "Add custom documents (.txt)",
        type=["txt"],
        accept_multiple_files=True,
        help=(
            f"Up to {MAX_UPLOAD_FILES} UTF-8 text files and "
            f"{MAX_UPLOAD_BYTES // 1_000} KB per file."
        ),
    )
    st.caption(
        f"CarWise limit: {MAX_UPLOAD_FILES} files, "
        f"{MAX_UPLOAD_BYTES // 1_000} KB each, 2 MB total."
    )
    if uploaded_files:
        corpus_choice = st.radio(
            "Search source",
            ["Uploaded text files", "Cambodian car listings"],
            help="Choose which document collection the RAG pipeline should search.",
        )
    else:
        corpus_choice = "Cambodian car listings"
    use_uploaded_corpus = corpus_choice == "Uploaded text files"
    st.caption(
        "Uploaded files stay in this Streamlit session and are not added to "
        "the saved CarWise dataset."
    )

try:
    prepared_documents, _ = ensure_cambodia_corpus(
        CAMBODIA_SNAPSHOT_CSV,
        CURATED_CSV,
        DOCUMENTS_DIR,
        specs_csv=REFERENCE_SPECS_CSV,
        images_csv=LISTING_IMAGES_CSV,
    )
    corpus_version = document_corpus_version(prepared_documents)
    documents, embedder, index = load_search_resources(
        DEFAULT_EMBEDDING_MODEL,
        corpus_version,
    )
except Exception as exc:
    st.error(f"CarWise could not start: {exc}")
    st.code(
        "python3 scripts/prepare_dataset.py\n"
        "python3 scripts/build_index.py\n"
        "python3 -m streamlit run app.py"
    )
    st.stop()

if use_uploaded_corpus:
    uploaded_payload = tuple(
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    )
    try:
        uploaded_documents = uploaded_text_documents(uploaded_payload)
        upload_version = document_corpus_version(uploaded_documents)
        if st.session_state.get("uploaded_corpus_version") != upload_version:
            with st.spinner("Chunking and indexing the uploaded documents..."):
                uploaded_chunks = chunk_documents(uploaded_documents)
                uploaded_index = VectorIndex.build(uploaded_chunks, embedder)
            st.session_state["uploaded_corpus_version"] = upload_version
            st.session_state["uploaded_corpus_resources"] = (
                uploaded_documents,
                uploaded_index,
            )
        documents, index = st.session_state["uploaded_corpus_resources"]
    except ValueError as exc:
        st.error(f"Uploaded corpus could not be used: {exc}")
        st.stop()

with st.sidebar:
    if use_uploaded_corpus:
        st.success("Uploaded corpus is active.")
        st.metric("Uploaded documents", len(documents))
        st.metric("Searchable chunks", len(index.chunks))
    else:
        years = sorted(
            {
                document.metadata.get("year")
                for document in documents
                if document.metadata.get("year")
            }
        )
        makes = sorted(
            {
                document.metadata.get("make")
                for document in documents
                if document.metadata.get("make")
            }
        )
        conditions = sorted(
            {
                document.metadata.get("condition")
                for document in documents
                if document.metadata.get("condition")
            }
        )
        provinces = sorted(
            {
                document.metadata.get("province")
                for document in documents
                if document.metadata.get("province")
            }
        )
        body_types = sorted(
            {
                document.metadata.get("body_type")
                for document in documents
                if document.metadata.get("body_type")
            }
        )
        prices = sorted(
            {
                int(document.metadata["price_usd"])
                for document in documents
                if document.metadata.get("price_usd")
            }
        )
        selected_years = st.multiselect("Model year", years, default=years)
        selected_makes = st.multiselect("Make", makes)
        selected_conditions = st.multiselect("Condition label", conditions)
        selected_provinces = st.multiselect("Province / municipality", provinces)
        selected_body_types = st.multiselect("Body type", body_types)
        maximum_price = st.slider(
            "Maximum asking price (USD)",
            min_value=(min(prices) // 500) * 500,
            max_value=((max(prices) + 499) // 500) * 500,
            value=((max(prices) + 499) // 500) * 500,
            step=500,
        )
        st.divider()
        st.metric("Cambodian listings", len(documents))
        st.metric("Searchable chunks", len(index.chunks))

examples = [
    "Which used SUVs in Phnom Penh are listed below $20,000?",
    "Recommend a Toyota Prius under $18,000.",
    "Find an electric car listed in Cambodia under $30,000.",
    "What is the safest car under $20,000?",
]


def choose_example(example_text: str) -> None:
    st.session_state["query_input"] = example_text


if use_uploaded_corpus:
    st.info(
        "Uploaded-corpus mode searches only the selected text files. "
        "Car filters and vehicle cards are disabled in this mode."
    )
    query = st.text_area(
        "Ask a question about the uploaded documents",
        placeholder="What do these documents say about the topic?",
        height=110,
        key="uploaded_query_input",
    )
else:
    query = st.text_area(
        "Describe the vehicle you need",
        placeholder=examples[0],
        height=110,
        key="query_input",
    )
    example_columns = st.columns(2)
    for index_number, example in enumerate(examples):
        example_columns[index_number % 2].button(
            example,
            key=f"example-{index_number}",
            use_container_width=True,
            on_click=choose_example,
            args=(example,),
        )

submitted = st.button(
    "Search uploaded documents" if use_uploaded_corpus else "Search and recommend",
    type="primary",
    use_container_width=True,
)
if submitted:
    if not query.strip():
        st.warning("Enter a question before searching.")
        st.stop()

    metadata_filters = None if use_uploaded_corpus else {
        "year": set(selected_years) if len(selected_years) < len(years) else set(),
        "make": set(selected_makes),
        "condition": set(selected_conditions),
        "province": set(selected_provinces),
        "body_type": set(selected_body_types),
        "price_usd": {price for price in prices if price <= maximum_price},
    }
    generator = make_generator(provider)
    pipeline = RAGPipeline(index, embedder, generator)
    started = time.perf_counter()
    with st.status("CarWise is searching...", expanded=True) as search_status:
        if use_uploaded_corpus:
            search_status.write(
                "Searching the uploaded documents, then using up to "
                f"{top_k} sources for the written answer."
            )
        else:
            search_status.write(
                "Searching all matching listings, then using up to "
                f"{top_k} sources for the written answer."
            )

        def show_search_step(stage: str, message: str) -> None:
            search_status.write(message)

        try:
            result = pipeline.answer(
                query,
                top_k=top_k,
                minimum_similarity=minimum_similarity,
                answer_mode=answer_mode,
                metadata_filters=metadata_filters,
                on_step=show_search_step,
                enforce_vehicle_scope=not use_uploaded_corpus,
                candidate_label=(
                    "uploaded documents" if use_uploaded_corpus else "vehicles"
                ),
            )
            if result.grounded:
                search_status.update(
                    label=(
                        "Search complete — grounded answer ready"
                        if use_uploaded_corpus
                        else "Search complete — grounded recommendation ready"
                    ),
                    state="complete",
                    expanded=True,
                )
            else:
                search_status.update(
                    label="Search complete — no supported answer produced",
                    state="complete",
                    expanded=True,
                )
        except GenerationError as exc:
            search_status.write(
                "The generation provider failed, so CarWise preserved the retrieved "
                "evidence instead of inventing an answer."
            )
            search_status.update(
                label="Search completed with a generation error",
                state="error",
                expanded=True,
            )
            st.error(f"Generation failed gracefully: {exc}")
            st.info(
                "The retrieved evidence is still shown below. Configure another provider "
                "or retry after checking the API."
            )
            result = None
        except requests.RequestException as exc:
            search_status.write(
                "The provider connection failed gracefully; retrieved evidence remains "
                "available below."
            )
            search_status.update(
                label="Search completed with a provider connection error",
                state="error",
                expanded=True,
            )
            st.error(f"Provider connection failed gracefully: {exc}")
            result = None

    elapsed = time.perf_counter() - started
    sources = result.sources if result else index.search(
        query,
        embedder,
        top_k=None,
        metadata_filters=metadata_filters,
    )
    if result is None:
        sources = [
            source for source in sources
            if source.score >= minimum_similarity
        ]

    if result and not result.grounded:
        st.warning(result.answer)
    elif not sources:
        st.info(
            "No uploaded documents matched the current question."
            if use_uploaded_corpus
            else "No vehicles matched the current question and filters."
        )
    else:
        matching_vehicles = (
            [] if use_uploaded_corpus else source_documents(sources, documents)
        )
        if matching_vehicles:
            st.subheader(f"Matching cars ({len(matching_vehicles)})")
            st.caption(
                "All qualifying Cambodian listings, ordered from strongest "
                "match to weakest."
            )
            render_vehicle_cards(matching_vehicles)

            target_price = matching_vehicles[0].metadata.get("price_usd")
            if isinstance(target_price, (int, float)):
                query_minimum, query_maximum = inferred_price_range(query)
                comparison_filters = dict(metadata_filters)
                comparison_filters["price_usd"] = {
                    price
                    for price in prices
                    if price <= maximum_price
                    and (query_minimum is None or price >= query_minimum)
                    and (query_maximum is None or price <= query_maximum)
                }
                target_fuel = matching_vehicles[0].metadata.get("fuel_type")
                if target_fuel and target_fuel != "Not reported":
                    comparison_filters["fuel_type"] = {target_fuel}
                target_body_type = matching_vehicles[0].metadata.get("body_type")
                if target_body_type and target_body_type != "Not reported":
                    comparison_filters["body_type"] = {target_body_type}
                similar_vehicles = similar_price_documents(
                    documents,
                    target_price,
                    exclude_document_ids={
                        vehicle.document_id for vehicle in matching_vehicles
                    },
                    metadata_filters=comparison_filters,
                    limit=3,
                )
                if similar_vehicles:
                    st.subheader(f"Cars around ${float(target_price):,.0f}")
                    st.caption(
                        "Other Cambodian listings with the closest asking prices."
                    )
                    render_vehicle_cards(similar_vehicles)

        if result and result.answer:
            st.subheader("Grounded answer")
            with st.container(border=True):
                # Streamlit interprets dollar-delimited text as LaTeX. Escape currency
                # symbols so LLM answers such as "$18,000" render as normal text.
                st.markdown(result.answer.replace("$", r"\$"))

    st.markdown(
        f"<div class='metric-row'>Provider: {generator.provider_name} · "
        f"Latency: {elapsed:.2f}s</div>",
        unsafe_allow_html=True,
    )

    if sources:
        with st.expander("View retrieval details and sources"):
            for source_number, source in enumerate(sources, start=1):
                st.markdown(
                    f"**[S{source_number}] {source.chunk.document_title}** · "
                    f"similarity {source.score:.3f}"
                )
                st.caption(
                    f"{source.chunk.source_name} · "
                    f"Section: {source.chunk.metadata.get('section', 'Unknown')}"
                )
                st.write(source.chunk.text)
                if source.chunk.source_url:
                    source_columns = st.columns(2)
                    source_columns[0].link_button(
                        "Open marketplace listing",
                        source.chunk.source_url,
                        use_container_width=True,
                    )
                    spec_source_url = source.chunk.metadata.get("spec_source_url")
                    if spec_source_url and spec_source_url != "Not verified":
                        source_columns[1].link_button(
                            "Open specification source",
                            spec_source_url,
                            use_container_width=True,
                        )
                else:
                    st.caption("Uploaded file — processed in this session only.")
                st.divider()
else:
    st.info(
        "Upload text files and ask a question to begin."
        if use_uploaded_corpus
        else "Enter a question or choose an example to begin."
    )
