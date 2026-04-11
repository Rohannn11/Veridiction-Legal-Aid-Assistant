"""Step 1: Local legal RAG retriever for Veridiction - Advanced Edition.

Builds and queries a local LlamaIndex vector store over comprehensive Indian legal datasets.
Implements TF-IDF keyword weighting, phrase matching, and domain-specific expansion.
English-only retrieval pipeline for MVP.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


LOGGER = logging.getLogger(__name__)


JUDGMENT_DATASETS: tuple[str, ...] = (
    "vihaannnn/Indian-Supreme-Court-Judgements-Chunked",
    "Subimal10/indian-legal-data-cleaned",
)
PROCEDURAL_DATASETS: tuple[str, ...] = (
    "viber1/indian-law-dataset",
    "ShreyasP123/Legal-Dataset-for-india",
    "nisaar/Lawyer_GPT_India",
)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROCEDURAL_INTENT_KEYWORDS: tuple[str, ...] = (
    "how",
    "which court",
    "where to file",
    "steps",
    "procedure",
    "process",
    "fees",
    "documents",
    "docs",
    "timeline",
    "time limit",
    "filing",
    "jurisdiction",
)

# Domain-specific keyword expansion for better matching
LEGAL_SYNONYMS: dict[str, list[str]] = {
    "wage": ["salary", "remuneration", "compensation", "payment", "dues", "stipend", "honorarium"],
    "unpaid": ["outstanding", "withheld", "denied", "owed", "pending"],
    "employer": ["company", "management", "employer", "proprietor", "firm"],
    "employee": ["worker", "staff", "worker", "laborer", "employee"],
    "domestic": ["conjugal", "family", "matrimonial", "household"],
    "violence": ["assault", "abuse", "threat", "harm", "injury", "brutality"],
    "property": ["land", "asset", "immovable", "estate", "real estate", "building"],
    "tenant": ["lessee", "occupier", "renter", "lodger"],
    "landlord": ["lessor", "owner", "proprietor"],
    "police": ["constable", "officer", "official", "authority"],
    "harassment": ["intimidation", "threat", "coercion", "abuse", "persecution"],
    "termination": ["dismissal", "termination", "discharge", "removal", "retrenchment"],
    "wrongful": ["unjustified", "improper", "unlawful", "illegal", "invalid"],
    "consumer": ["customer", "buyer", "purchaser", "consumer"],
    "fraud": ["deception", "cheating", "misrepresentation", "breach", "falsehood"],
}


class RetrieverError(RuntimeError):
    """Raised when the retriever cannot build or query the index."""


@dataclass(slots=True)
class RetrieverConfig:
    """Runtime configuration for building and querying the retriever."""

    dataset_ids: tuple[str, ...] = JUDGMENT_DATASETS
    procedural_dataset_ids: tuple[str, ...] = PROCEDURAL_DATASETS
    split: str = "train"
    top_k: int = 5
    embedding_model_name: str = DEFAULT_EMBED_MODEL
    persist_dir: Path = Path("data") / "vector_index"
    judgment_persist_dir: Path = Path("data") / "vector_index"
    hf_cache_dir: Path = Path("data") / "hf_cache"
    hf_token: str | None = None
    min_confidence: float = 0.20
    keyword_boost: float = 0.35  # Base keyword boost weight
    phrase_boost: float = 0.50   # Bonus for phrase matches
    synonym_boost: float = 0.15  # Bonus for synonym matches
    tfidf_enabled: bool = True   # Use TF-IDF weighting
    procedural_streaming: bool = True  # Stream procedural datasets to reduce local disk usage
    procedural_max_documents: int = 3500  # Limit memory usage for procedural corpus
    max_rewrites_procedural: int = 2
    max_rewrites_substantive: int = 2
    procedural_fallback_threshold: float = 0.84
    procedural_fallback_top_k: int = 2


class LegalRetriever:
    """Legal passage retriever backed by LlamaIndex VectorStoreIndex with advanced keyword boosting."""

    _TEXT_KEYS: tuple[str, ...] = (
        "text",
        "chunk",
        "passage",
        "content",
        "clean_text",
        "judgment_text",
        "body",
    )

    _METADATA_KEYS: tuple[str, ...] = (
        "source",
        "dataset",
        "case_name",
        "citation",
        "year",
        "court",
        "url",
        "doc_id",
        "chunk_id",
        "title",
    )

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        if self.config.persist_dir != Path("data") / "vector_index":
            self.config.judgment_persist_dir = self.config.persist_dir
        self._index: VectorStoreIndex | None = None
        self._judgment_index: VectorStoreIndex | None = None
        self._idf_dict: dict[str, float] = {}  # Cached IDF values
        self._all_documents: list[Document] = []  # For IDF calculation
        self._procedural_corpus: list[dict[str, Any]] | None = None
        token = self._get_hf_token()
        if token and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = token
        self._embed_model = HuggingFaceEmbedding(model_name=self.config.embedding_model_name)

    def load_or_build_index(
        self,
        force_rebuild: bool = False,
        max_documents: int | None = None,
    ) -> VectorStoreIndex:
        """Load an existing index from disk or build one from datasets."""
        persist_dir = self.config.judgment_persist_dir
        if persist_dir.exists() and not force_rebuild:
            LOGGER.info("Loading existing vector index from %s", persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self._judgment_index = load_index_from_storage(storage_context, embed_model=self._embed_model)
            self._index = self._judgment_index
            return self._judgment_index

        return self.build_index(max_documents=max_documents)

    def build_index(self, max_documents: int | None = None) -> VectorStoreIndex:
        """Build and persist a fresh vector index from configured datasets."""
        persist_dir = self.config.judgment_persist_dir
        hf_cache_dir = self.config.hf_cache_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        documents = list(self._load_documents(max_documents=max_documents))
        if not documents:
            raise RetrieverError("No documents were loaded from configured datasets.")

        self._all_documents = documents
        
        # Calculate IDF values for all documents
        if self.config.tfidf_enabled:
            LOGGER.info("Calculating IDF values for %d documents", len(documents))
            self._calculate_idf(documents)

        LOGGER.info("Creating vector index with %d documents", len(documents))
        self._judgment_index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self._embed_model,
            show_progress=True,
        )
        self._judgment_index.storage_context.persist(str(persist_dir))
        self._index = self._judgment_index
        LOGGER.info("Index persisted to %s", persist_dir)
        return self._judgment_index
    
    def _calculate_idf(self, documents: list[Document]) -> None:
        """Calculate inverse document frequency for all keywords in the corpus."""
        total_docs = len(documents)
        doc_frequencies: dict[str, int] = Counter()
        
        for doc in documents:
            # Access text from Document object correctly
            text = doc.text if hasattr(doc, 'text') else str(doc)
            text = text.lower()
            words = set(re.findall(r"\b\w+\b", text))
            for word in words:
                if len(word) > 2:  # Skip short words
                    doc_frequencies[word] += 1
        
        # Calculate IDF: log(total_docs / doc_frequency)
        for word, freq in doc_frequencies.items():
            self._idf_dict[word] = math.log(total_docs / (1 + freq))
        
        LOGGER.info("Calculated IDF for %d unique terms", len(self._idf_dict))

    def query(self, user_query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k passages from routed dual-index retrieval (judgment + procedural)."""
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty.")

        retrieval_start = time.perf_counter()

        similarity_top_k = top_k if top_k is not None else self.config.top_k
        procedural_intent = self._is_procedural_intent(user_query)
        rewritten_queries = self._rewrite_queries(user_query, procedural_intent=procedural_intent)

        if procedural_intent:
            procedural_k = max(2, similarity_top_k)
            judgment_k = max(2, similarity_top_k // 2)
        else:
            procedural_k = max(1, similarity_top_k // 2)
            judgment_k = max(2, similarity_top_k)

        judgment_results: list[dict[str, Any]] = []
        procedural_results: list[dict[str, Any]] = []

        for idx, rewritten in enumerate(rewritten_queries):
            query_bias = 0.04 if idx == 0 else 0.0
            for item in self._query_judgment_index(user_query=rewritten, top_k=judgment_k):
                with_bias = dict(item)
                with_bias["score"] = min(1.0, float(item.get("score", 0.0)) + query_bias)
                with_bias["rewrite_query"] = rewritten
                with_bias["rewrite_rank"] = idx
                judgment_results.append(with_bias)

            if procedural_intent:
                for item in self._query_procedural_index(user_query=rewritten, top_k=procedural_k):
                    with_bias = dict(item)
                    with_bias["score"] = min(1.0, float(item.get("score", 0.0)) + query_bias)
                    with_bias["rewrite_query"] = rewritten
                    with_bias["rewrite_rank"] = idx
                    procedural_results.append(with_bias)

        procedural_fallback_used = False
        if not procedural_intent:
            best_judgment_score = max((float(i.get("score", 0.0)) for i in judgment_results), default=0.0)
            should_probe_procedural = not judgment_results or best_judgment_score < self.config.procedural_fallback_threshold
            if should_probe_procedural:
                procedural_fallback_used = True
                fallback_query = rewritten_queries[0] if rewritten_queries else user_query
                fallback_top_k = max(1, min(similarity_top_k, self.config.procedural_fallback_top_k))
                for item in self._query_procedural_index(user_query=fallback_query, top_k=fallback_top_k):
                    with_bias = dict(item)
                    with_bias["rewrite_query"] = fallback_query
                    with_bias["rewrite_rank"] = 0
                    procedural_results.append(with_bias)

        merged = self._merge_dual_results(
            query=user_query,
            judgment_results=judgment_results,
            procedural_results=procedural_results,
            top_k=similarity_top_k,
            procedural_intent=procedural_intent,
            rewritten_queries=rewritten_queries,
        )

        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000
        for item in merged:
            metadata = dict(item.get("metadata", {}) or {})
            metadata["retrieval_elapsed_ms"] = round(retrieval_elapsed_ms, 2)
            metadata["procedural_fallback_used"] = procedural_fallback_used
            item["metadata"] = metadata

        return merged

    def _is_procedural_intent(self, query: str) -> bool:
        lowered = query.lower()
        return any(token in lowered for token in PROCEDURAL_INTENT_KEYWORDS)

    def _rewrite_queries(self, query: str, procedural_intent: bool) -> list[str]:
        base = " ".join(query.split()).strip()
        if not base:
            return [query]

        lowered = base.lower()
        rewrites = [base]

        if procedural_intent:
            rewrites.append(f"{base}. What are the filing steps, forum, and required documents in Maharashtra?")
        else:
            rewrites.append(f"{base}. Relevant Indian legal precedents and remedies.")

        if "maharashtra" not in lowered:
            rewrites.append(f"Maharashtra legal process for: {base}")

        if not any(token in lowered for token in ("document", "evidence", "proof")):
            rewrites.append(f"Evidence and documents needed for: {base}")

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in rewrites:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        max_rewrites = self.config.max_rewrites_procedural if procedural_intent else self.config.max_rewrites_substantive
        return unique[:max(1, max_rewrites)]

    def _query_judgment_index(self, user_query: str, top_k: int) -> list[dict[str, Any]]:
        if self._judgment_index is None:
            self.load_or_build_index(force_rebuild=False)

        if self._judgment_index is None:
            raise RetrieverError("Judgment index could not be initialized.")

        retriever = self._judgment_index.as_retriever(similarity_top_k=max(1, top_k) * 5)
        nodes = retriever.retrieve(user_query)

        keywords = self._extract_keywords_advanced(user_query)
        phrases = self._extract_phrases(user_query)
        expanded_keywords = self._expand_with_synonyms(keywords)

        boosted_nodes = self._boost_by_keywords_advanced(
            nodes,
            keywords,
            phrases,
            expanded_keywords,
        )
        final_nodes = sorted(boosted_nodes, key=lambda n: float(n.score or 0.0), reverse=True)[:max(1, top_k)]
        final_nodes = self._calibrate_scores(final_nodes)

        results: list[dict[str, Any]] = []
        for node in final_nodes:
            text = node.node.text if hasattr(node.node, "text") else str(node.node)
            text = text.strip() if text else ""
            metadata = dict(getattr(node.node, "metadata", {}) or {})
            metadata["source_index"] = "judgment_index"
            metadata["source_label"] = "Judgment Index"
            score = float(node.score) if node.score is not None else 0.0
            results.append({"passage": text, "metadata": metadata, "score": score})
        return results

    def _query_procedural_index(self, user_query: str, top_k: int) -> list[dict[str, Any]]:
        corpus = self._load_procedural_corpus(max_documents=self.config.procedural_max_documents)
        if not corpus:
            return []

        keywords = self._extract_keywords_advanced(user_query)
        phrases = self._extract_phrases(user_query)
        expanded = self._expand_with_synonyms(keywords)
        keyword_patterns = [re.compile(rf"\b{re.escape(keyword)}\b") for keyword in keywords]
        synonym_terms: set[str] = set()
        for synonyms in expanded.values():
            for synonym in synonyms[1:]:
                synonym_terms.add(synonym)
        synonym_patterns = [re.compile(rf"\b{re.escape(synonym)}\b") for synonym in synonym_terms]

        scored: list[dict[str, Any]] = []
        for row in corpus:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            lowered = text.lower()
            score = 0.0

            for phrase in phrases:
                if phrase in lowered:
                    score += self.config.phrase_boost * 1.1

            for pattern in keyword_patterns:
                if pattern.search(lowered):
                    score += self.config.keyword_boost

            for pattern in synonym_patterns:
                if pattern.search(lowered):
                    score += self.config.synonym_boost

            if score <= 0:
                continue

            metadata = dict(row.get("metadata", {}) or {})
            metadata["source_index"] = "procedural_index"
            metadata["source_label"] = "Procedural Index"
            scored.append({"passage": text, "metadata": metadata, "score": min(score, 1.0)})

        scored = sorted(scored, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return scored[:max(1, top_k)]

    def _merge_dual_results(
        self,
        query: str,
        judgment_results: list[dict[str, Any]],
        procedural_results: list[dict[str, Any]],
        top_k: int,
        procedural_intent: bool,
        rewritten_queries: list[str],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        source_counts: dict[str, int] = {"judgment_index": 0, "procedural_index": 0}

        def _add(items: list[dict[str, Any]], source_bias: float = 0.0) -> None:
            for item in items:
                passage = str(item.get("passage", "")).strip()
                if not passage:
                    continue
                key = " ".join(passage.lower().split())[:320]
                if key in seen:
                    continue
                seen.add(key)
                score = float(item.get("score", 0.0)) + source_bias
                metadata = dict(item.get("metadata", {}) or {})
                source_index = str(metadata.get("source_index", "judgment_index"))
                if source_counts.get(source_index, 0) >= max(1, top_k // 2):
                    score -= 0.07
                merged.append({
                    "passage": passage,
                    "metadata": metadata,
                    "score": min(score, 1.0),
                    "rewrite_query": str(item.get("rewrite_query", query)),
                    "rewrite_rank": int(item.get("rewrite_rank", 0)),
                })
                source_counts[source_index] = source_counts.get(source_index, 0) + 1

        if procedural_intent:
            _add(procedural_results, source_bias=0.08)
            _add(judgment_results, source_bias=0.0)
        else:
            _add(judgment_results, source_bias=0.05)
            _add(procedural_results, source_bias=0.0)

        merged = self._rerank_with_diversity(merged, top_k=max(1, top_k))

        route_label = "procedural_priority" if procedural_intent else "judgment_priority"
        query_variants = "; ".join(rewritten_queries)
        for item in merged:
            metadata = dict(item.get("metadata", {}) or {})
            metadata["retrieval_route"] = route_label
            metadata["query_variants"] = query_variants
            item["metadata"] = metadata

        merged = sorted(merged, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:max(1, top_k)]

    def _rerank_with_diversity(self, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not items:
            return items

        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        selected: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()

        for item in ranked:
            passage = str(item.get("passage", "")).lower()
            words = [w for w in re.findall(r"\b\w+\b", passage) if len(w) > 3]
            signature = " ".join(words[:10])
            if signature and signature in seen_signatures:
                continue

            metadata = item.get("metadata", {}) or {}
            source = str(metadata.get("source_index", ""))
            if source:
                same_source = sum(
                    1 for s in selected if str((s.get("metadata", {}) or {}).get("source_index", "")) == source
                )
                if same_source >= max(1, top_k // 2):
                    continue

            selected.append(item)
            if signature:
                seen_signatures.add(signature)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            selected_keys = {id(x) for x in selected}
            for item in ranked:
                if id(item) in selected_keys:
                    continue
                selected.append(item)
                if len(selected) >= top_k:
                    break

        return selected
    
    def _calibrate_scores(self, nodes: list) -> list:
        """Normalize scores to 0.80-0.95 range for high-quality results."""
        if not nodes:
            return nodes
        
        scores = [float(n.score or 0.0) for n in nodes]
        if not scores:
            return nodes
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Map scores from [min_score, max_score] to [0.80, 0.95]
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for node in nodes:
            old_score = float(node.score or 0.0)
            # Linear interpolation
            normalized = (old_score - min_score) / score_range if score_range > 0 else 0.5
            # Map to 0.80-0.95 range
            new_score = 0.80 + (normalized * 0.15)
            node.score = new_score
        
        return nodes

    def _extract_keywords_advanced(self, query: str) -> list[str]:
        """Extract key terms from query, excluding stop words and short words."""
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "for", "of", "to", "is", "am", "are",
            "my", "have", "has", "been", "be", "not", "do", "does", "did", "will", "would",
            "been", "having", "should", "could", "must", "may", "might", "can", "i", "you",
            "he", "she", "it", "we", "they", "that", "this", "these", "those", "which", "who",
            "when", "where", "why", "how", "what", "if", "from", "at", "on", "by", "with",
            "as", "about", "into", "through", "during", "before", "after", "above", "below",
        }
        words = re.findall(r"\b\w+\b", query.lower())
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    def _extract_phrases(self, query: str) -> list[str]:
        """Extract multi-word phrases from query (2-3 word phrases)."""
        words = query.lower().split()
        phrases = []
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}".strip()
            if len(phrase) > 5 and not any(sw in phrase for sw in ["the", "and", "but", "for"]):
                phrases.append(phrase)
        
        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}".strip()
            if len(phrase) > 10:
                phrases.append(phrase)
        
        return phrases
    
    def _expand_with_synonyms(self, keywords: list[str]) -> dict[str, list[str]]:
        """Expand keywords with legal synonyms for better matching."""
        expanded = {}
        for keyword in keywords:
            expanded[keyword] = [keyword]
            # Check if keyword matches any legal synonym keys
            for term, synonyms in LEGAL_SYNONYMS.items():
                if keyword in synonyms or term == keyword:
                    expanded[keyword].extend(synonyms)
                    break
        return expanded

    def _boost_by_keywords_advanced(
        self,
        nodes: list,
        keywords: list[str],
        phrases: list[str],
        expanded_keywords: dict[str, list[str]],
    ) -> list:
        """Apply advanced keyword boosting using TF-IDF, phrases, and synonyms."""
        if not keywords:
            return nodes
        
        for node in nodes:
            # Access text from node correctly
            node_text = node.node.text if hasattr(node.node, 'text') else str(node.node)
            text = node_text.lower() if node_text else ""
            original_score = float(node.score or 0.0)
            boost = 0.0
            
            # 1. Phrase matching (highest priority)
            for phrase in phrases:
                if phrase in text:
                    phrase_match_count = len(re.findall(re.escape(phrase), text))
                    boost += phrase_match_count * self.config.phrase_boost
            
            # 2. TF-IDF keyword boosting (medium priority)
            for keyword in keywords:
                if keyword in text:
                    keyword_count = len(re.findall(rf"\b{re.escape(keyword)}\b", text))
                    # Use IDF weighting if available
                    idf = self._idf_dict.get(keyword, 1.0) if self.config.tfidf_enabled else 1.0
                    idf = max(idf, 0.5)  # Clamp to avoid very low values
                    weighted_boost = keyword_count * self.config.keyword_boost * idf
                    boost += weighted_boost
            
            # 3. Synonym matching (lower priority)
            for keyword, synonyms in expanded_keywords.items():
                for synonym in synonyms[1:]:  # Skip first (the keyword itself)
                    if synonym in text:
                        synonym_count = len(re.findall(rf"\b{re.escape(synonym)}\b", text))
                        boost += synonym_count * self.config.synonym_boost
            
            # Apply boost with cap at 0.5
            boost = min(boost, 0.5)
            new_score = min(original_score + boost, 1.0)
            node.score = new_score
        
        return nodes

    def _load_documents(self, max_documents: int | None = None) -> Iterable[Document]:
        """Load and normalize dataset rows into LlamaIndex documents."""
        loaded = 0
        seen_texts: set[str] = set()
        token = self._get_hf_token()

        for dataset_id in self.config.dataset_ids:
            LOGGER.info("Loading dataset: %s", dataset_id)
            dataset_obj = load_dataset(
                dataset_id,
                split=self.config.split,
                token=token,
                cache_dir=str(self.config.hf_cache_dir),
            )

            dataset = self._resolve_dataset(dataset_obj, self.config.split)
            for row in dataset:
                text = self._extract_text(row)
                if not text:
                    continue

                normalized = " ".join(text.split())
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)

                metadata = self._extract_metadata(row)
                metadata["dataset"] = dataset_id
                yield Document(text=text, metadata=metadata)

                loaded += 1
                if max_documents is not None and loaded >= max_documents:
                    LOGGER.info("Reached max_documents=%d", max_documents)
                    return

    def _load_procedural_corpus(self, max_documents: int | None = None) -> list[dict[str, Any]]:
        """Load procedural corpus in streaming mode to avoid large local disk usage."""
        if self._procedural_corpus is not None:
            return self._procedural_corpus

        token = self._get_hf_token()
        max_docs = max_documents if max_documents is not None else self.config.procedural_max_documents
        dataset_count = max(1, len(self.config.procedural_dataset_ids))
        per_dataset_limit = max(150, max_docs // dataset_count)

        corpus: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for dataset_id in self.config.procedural_dataset_ids:
            LOGGER.info("Loading procedural dataset (streaming=%s): %s", self.config.procedural_streaming, dataset_id)
            dataset_obj = load_dataset(
                dataset_id,
                split=self.config.split,
                token=token,
                cache_dir=str(self.config.hf_cache_dir),
                streaming=self.config.procedural_streaming,
            )

            loaded_for_dataset = 0
            for row in dataset_obj:
                text = self._extract_text(row)
                if not text:
                    continue

                normalized = " ".join(text.split())
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)

                metadata = self._extract_metadata(row)
                metadata["dataset"] = dataset_id
                corpus.append({"text": text, "metadata": metadata})
                loaded_for_dataset += 1

                if loaded_for_dataset >= per_dataset_limit:
                    break
                if len(corpus) >= max_docs:
                    break

            if len(corpus) >= max_docs:
                break

        self._procedural_corpus = corpus
        LOGGER.info("Procedural corpus loaded in-memory with %d passages", len(corpus))
        return corpus

    def _resolve_dataset(self, dataset_obj: Dataset | DatasetDict, split: str) -> Dataset:
        """Support both Dataset and DatasetDict returns from load_dataset."""
        if isinstance(dataset_obj, Dataset):
            return dataset_obj
        if split in dataset_obj:
            return dataset_obj[split]
        first_split = next(iter(dataset_obj.keys()), None)
        if first_split is None:
            raise RetrieverError("Dataset loaded but no split found.")
        return dataset_obj[first_split]

    def _extract_text(self, row: dict[str, Any]) -> str | None:
        """Find first non-empty text-like field in a row."""
        for key in self._TEXT_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for value in row.values():
            if isinstance(value, str) and len(value.strip()) > 100:
                return value.strip()
        return None

    def _extract_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        """Collect metadata fields useful for traceability and citation."""
        metadata: dict[str, Any] = {}
        for key in self._METADATA_KEYS:
            value = row.get(key)
            if value is not None and value != "":
                metadata[key] = value
        return metadata

    def _get_hf_token(self) -> str | None:
        """Resolve HF token from config, environment, then local .env file."""
        if self.config.hf_token:
            return self.config.hf_token

        env_token = os.getenv("HUGGINGFACE_TOKEN")
        if env_token:
            return env_token.strip()

        env_path = Path(".env")
        if not env_path.exists():
            return None

        content = env_path.read_text(encoding="utf-8")
        match = re.search(r"^\s*HUGGINGFACE_TOKEN\s*=\s*(.+?)\s*$", content, flags=re.MULTILINE)
        if not match:
            return None

        # Support plain values and quoted values.
        raw = match.group(1).strip().strip('"').strip("'")
        return raw or None


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query Veridiction legal retriever.")
    parser.add_argument("--query", type=str, default="unpaid wages", help="Query text")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Limit docs while building index for quick smoke tests",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild index from dataset",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_cli().parse_args()
    retriever = LegalRetriever()
    retriever.load_or_build_index(
        force_rebuild=args.force_rebuild,
        max_documents=args.max_documents,
    )
    results = retriever.query(args.query, top_k=args.top_k)

    print(f"Top {len(results)} results for query: {args.query}")
    for idx, item in enumerate(results, start=1):
        preview = item["passage"][:280].replace("\n", " ")
        print(f"[{idx}] score={item['score']}\n{preview}\nmetadata={item['metadata']}\n")


if __name__ == "__main__":
    main()