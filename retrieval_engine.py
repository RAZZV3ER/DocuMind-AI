import re
import math
from collections import Counter
from functools import lru_cache

# Optional semantic retrieval.
# If Windows blocks PyTorch, keyword retrieval will still work.
try:
    from sentence_transformers import SentenceTransformer
except (ImportError, OSError):
    SentenceTransformer = None


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
RRF_K = 60

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the semantic embedding model."""
    if SentenceTransformer is None:
        raise RuntimeError(
            "Semantic retrieval is unavailable. "
            "Using keyword retrieval instead."
        )

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    """Normalize whitespace."""
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalize_for_matching(text):
    """Normalize text for evidence comparison."""
    return normalize_text(text).casefold()


def safe_int(value, default):
    """Convert a value to an integer safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tokenize(text):
    """Tokenize text without requiring scikit-learn."""
    return re.findall(r"(?u)\b\w+\b", text.casefold())


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def chunk_document(document_text):
    """Split extracted document text into overlapping chunks."""

    if not isinstance(document_text, str) or not document_text.strip():
        return []

    lines = document_text.splitlines()

    chunks = []
    current_lines = []
    current_length = 0
    chunk_id = 0

    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()

        if not line:
            continue

        # Handle lines longer than the chunk limit.
        if len(line) > CHUNK_SIZE:
            if current_lines:
                chunk_id += 1
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": "\n".join(
                        item[1] for item in current_lines
                    ),
                    "start_line": current_lines[0][0],
                    "end_line": current_lines[-1][0],
                })

                current_lines = []
                current_length = 0

            for start in range(0, len(line), CHUNK_SIZE):
                piece = line[start:start + CHUNK_SIZE]
                chunk_id += 1

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": piece,
                    "start_line": line_number,
                    "end_line": line_number,
                })

            continue

        required_length = len(line) + (1 if current_lines else 0)

        if (
            current_lines
            and current_length + required_length > CHUNK_SIZE
        ):
            chunk_id += 1

            chunks.append({
                "chunk_id": chunk_id,
                "text": "\n".join(
                    item[1] for item in current_lines
                ),
                "start_line": current_lines[0][0],
                "end_line": current_lines[-1][0],
            })

            # Retain recent lines for overlapping context.
            overlap_lines = []
            overlap_length = 0

            for item in reversed(current_lines):
                item_length = len(item[1]) + 1

                if (
                    overlap_lines
                    and overlap_length + item_length > CHUNK_OVERLAP
                ):
                    break

                if (
                    item_length > CHUNK_OVERLAP
                    and not overlap_lines
                ):
                    continue

                overlap_lines.insert(0, item)
                overlap_length += item_length

            current_lines = overlap_lines
            current_length = sum(
                len(item[1]) + 1 for item in current_lines
            )

        current_lines.append((line_number, line))
        current_length += len(line) + 1

    if current_lines:
        chunk_id += 1

        chunks.append({
            "chunk_id": chunk_id,
            "text": "\n".join(
                item[1] for item in current_lines
            ),
            "start_line": current_lines[0][0],
            "end_line": current_lines[-1][0],
        })

    return chunks


# ============================================================
# PURE-PYTHON TF-IDF
# ============================================================

def build_tfidf_vector(tokens, document_frequency, total_documents):
    """Build a sparse TF-IDF vector using pure Python."""

    term_counts = Counter(tokens)
    vector = {}

    for term, count in term_counts.items():
        # Sublinear term frequency.
        tf = 1.0 + math.log(count)

        # Smoothed inverse document frequency.
        df = document_frequency.get(term, 0)
        idf = math.log(
            (1 + total_documents) / (1 + df)
        ) + 1.0

        vector[term] = tf * idf

    return vector


def sparse_cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between sparse dictionaries."""

    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(
        value * vector_b.get(term, 0.0)
        for term, value in vector_a.items()
    )

    norm_a = math.sqrt(
        sum(value * value for value in vector_a.values())
    )

    norm_b = math.sqrt(
        sum(value * value for value in vector_b.values())
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ============================================================
# KEYWORD RETRIEVAL
# ============================================================

def keyword_rank(query, chunks):
    """
    Rank chunks using TF-IDF unigram and bigram features.
    No scikit-learn dependency is required.
    """

    if not chunks or not normalize_text(query):
        return []

    document_tokens = []

    for chunk in chunks:
        tokens = tokenize(chunk["text"])

        # Include unigrams and adjacent bigrams.
        features = list(tokens)
        features.extend(
            f"{tokens[i]} {tokens[i + 1]}"
            for i in range(len(tokens) - 1)
        )

        document_tokens.append(features)

    query_tokens = tokenize(query)
    query_features = list(query_tokens)
    query_features.extend(
        f"{query_tokens[i]} {query_tokens[i + 1]}"
        for i in range(len(query_tokens) - 1)
    )

    if not query_features:
        return list(range(len(chunks)))

    # Calculate document frequency for each feature.
    document_frequency = Counter()

    for features in document_tokens:
        document_frequency.update(set(features))

    total_documents = len(document_tokens)

    query_vector = build_tfidf_vector(
        query_features,
        document_frequency,
        total_documents,
    )

    scores = []

    for index, features in enumerate(document_tokens):
        document_vector = build_tfidf_vector(
            features,
            document_frequency,
            total_documents,
        )

        score = sparse_cosine_similarity(
            query_vector,
            document_vector,
        )

        scores.append((index, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    return [index for index, score in scores]


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_rank(query, chunks):
    """Rank chunks using SentenceTransformer embeddings."""

    if not chunks or not normalize_text(query):
        return []

    if SentenceTransformer is None:
        raise RuntimeError(
            "SentenceTransformer is unavailable."
        )

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Normalized embeddings allow cosine similarity via dot product.
    scores = (embeddings @ query_embedding[0]).tolist()

    return sorted(
        range(len(chunks)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(rankings, rrf_k=RRF_K):
    """Combine rankings using Reciprocal Rank Fusion."""

    scores = {}

    for ranking in rankings:
        for rank, item_index in enumerate(ranking, start=1):
            scores[item_index] = (
                scores.get(item_index, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    ranked_indices = sorted(
        scores,
        key=lambda index: scores[index],
        reverse=True,
    )

    return ranked_indices, scores


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query, document_text, top_k=TOP_K):
    """
    Perform hybrid retrieval using TF-IDF and semantic search.

    If semantic retrieval is blocked or fails, use keyword retrieval.
    """

    query = normalize_text(query)

    if not query:
        return []

    chunks = chunk_document(document_text)

    if not chunks:
        return []

    top_k = max(1, safe_int(top_k, TOP_K))

    keyword_results = keyword_rank(query, chunks)

    semantic_available = False
    semantic_results = []

    try:
        semantic_results = semantic_rank(query, chunks)
        semantic_available = bool(semantic_results)

    except Exception:
        # Preserve functionality if semantic retrieval fails.
        semantic_results = []

    rankings = [keyword_results]

    if semantic_available:
        rankings.append(semantic_results)

    ranked_indices, rrf_scores = reciprocal_rank_fusion(rankings)

    keyword_positions = {
        index: rank
        for rank, index in enumerate(keyword_results, start=1)
    }

    semantic_positions = {
        index: rank
        for rank, index in enumerate(semantic_results, start=1)
    }

    results = []
    seen_texts = set()

    for index in ranked_indices:
        chunk = chunks[index]
        normalized_chunk = normalize_for_matching(chunk["text"])

        if normalized_chunk in seen_texts:
            continue

        seen_texts.add(normalized_chunk)

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "score": round(rrf_scores[index], 6),
            "retrieval_mode": (
                "hybrid"
                if semantic_available
                else "keyword_fallback"
            ),
            "keyword_rank": keyword_positions.get(index),
            "semantic_rank": semantic_positions.get(index),
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# EVIDENCE QUOTE VERIFICATION
# ============================================================

def verify_evidence_quote(quote, passages):
    """Check whether a quote appears in retrieved passages."""

    normalized_quote = normalize_for_matching(quote)

    if not normalized_quote:
        return {
            "verified": False,
            "message": "No quote provided.",
            "sources": [],
        }

    sources = []

    for passage in passages or []:
        passage_text = normalize_for_matching(
            passage.get("text", "")
        )

        if normalized_quote in passage_text:
            sources.append({
                "chunk_id": passage.get("chunk_id"),
                "start_line": passage.get("start_line"),
                "end_line": passage.get("end_line"),
                "text": passage.get("text", ""),
            })

    return {
        "verified": bool(sources),
        "message": (
            "Exact quote found in retrieved evidence."
            if sources
            else "Quote not found in retrieved evidence."
        ),
        "sources": sources,
    }


# ============================================================
# SOURCE-BASED ANSWER EVIDENCE
# ============================================================

def build_evidence_context(results):
    """Create evidence context for an LLM prompt."""

    if not results:
        return "No relevant evidence was retrieved."

    sections = []

    for number, result in enumerate(results, start=1):
        sections.append(
            f"[Source {number} | Chunk "
            f"{result.get('chunk_id', '?')} "
            f"| Extracted-text lines "
            f"{result.get('start_line', '?')}-"
            f"{result.get('end_line', '?')}]\n"
            f"{result.get('text', '')}"
        )

    return "\n\n".join(sections)


# ============================================================
# FORMAT SEARCH RESULTS
# ============================================================

def format_search_results(results):
    """Format retrieved passages with source references."""

    if not results:
        return "No relevant passages were found in the document."

    formatted = []

    for number, result in enumerate(results, start=1):
        start_line = result.get("start_line", "?")
        end_line = result.get("end_line", "?")
        chunk_id = result.get("chunk_id", "?")
        score = result.get("score", 0.0)
        text = result.get("text", "")

        formatted.append(
            f"[Passage {number} | Chunk {chunk_id} | "
            f"Extracted-text lines {start_line}-{end_line} | "
            f"RRF score {score}]\n{text}"
        )

    return "\n\n".join(formatted)


# ============================================================
# DOCUMENT RETRIEVAL HELPER
# ============================================================

def search_document(query, document_text, top_k=TOP_K):
    """Return retrieval results and formatted evidence context."""

    results = hybrid_search(
        query=query,
        document_text=document_text,
        top_k=top_k,
    )

    return {
        "query": normalize_text(query),
        "results": results,
        "evidence_context": build_evidence_context(results),
        "formatted_results": format_search_results(results),
        "result_count": len(results),
    }