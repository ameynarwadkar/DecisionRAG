import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import litellm
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.config import Settings
from dec_rag.models import DocumentPage, TopicCluster

VERBATIM_CLUSTER_SUMMARY_PROMPT = (
    "Write one sentence (at most 30 words) describing what the pages in this group are about. "
    "Name the main kinds of subjects it covers, including the less typical ones, so that a reader "
    "can tell whether a given Wikipedia page would belong to this group."
)

EMBEDDING_BATCH_SIZE = 128
MAX_TITLES_FOR_SUMMARY = 300


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return mat / norms


def cosine_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    max_iters: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical (cosine) k-means with k-means++ seeding."""
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("Cannot cluster zero vectors.")
    k = max(1, min(k, n))
    x = _l2_normalize(vectors.astype(np.float64))

    rng = np.random.default_rng(seed)
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    first_idx = int(rng.integers(0, n))
    centroids[0] = x[first_idx]

    # Cosine distance = 1 - cos_sim in [0, 2]
    min_dists = np.maximum(0.0, 1.0 - (x @ centroids[0]))
    for c_idx in range(1, k):
        total = float(min_dists.sum())
        if total <= 1e-12:
            pick = int(rng.integers(0, n))
        else:
            probs = min_dists / total
            pick = int(rng.choice(n, p=probs))
        centroids[c_idx] = x[pick]
        new_dists = np.maximum(0.0, 1.0 - (x @ centroids[c_idx]))
        min_dists = np.minimum(min_dists, new_dists)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iters):
        sims = x @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels

        for c_idx in range(k):
            members = x[labels == c_idx]
            if len(members) == 0:
                # Re-seed empty cluster with point farthest from its assigned centroid
                assigned_sims = sims[np.arange(n), labels]
                farthest = int(np.argmin(assigned_sims))
                centroids[c_idx] = x[farthest]
                labels[farthest] = c_idx
            else:
                mean_vec = members.mean(axis=0, keepdims=True)
                centroids[c_idx] = _l2_normalize(mean_vec)[0]

    return labels, centroids


def _cap_thirty_words(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    words = cleaned.split(" ")
    if len(words) <= 30:
        return cleaned
    capped = " ".join(words[:30]).rstrip(",;:")
    if capped and capped[-1] not in ".!?":
        capped += "."
    return capped


def _extract_embedding_list(resp: Any) -> list[list[float]]:
    data = resp["data"] if isinstance(resp, dict) else resp.data
    vectors: list[list[float]] = []
    for item in data:
        emb = item["embedding"] if isinstance(item, dict) else item.embedding
        vectors.append([float(v) for v in emb])
    return vectors


def _extract_completion_text(resp: Any) -> str:
    choices = resp["choices"] if isinstance(resp, dict) else resp.choices
    first = choices[0]
    msg = first["message"] if isinstance(first, dict) else first.message
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return str(content or "").strip()


async def _embed_page_openings(
    texts: Sequence[str],
    model: str,
    api_key: str | None = None,
) -> np.ndarray:
    extra: dict[str, Any] = {"api_key": api_key.strip()} if (api_key and api_key.strip()) else {}
    all_vecs: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
        resp = await litellm.aembedding(model=model, input=batch, **extra)
        all_vecs.extend(_extract_embedding_list(resp))
    return np.asarray(all_vecs, dtype=np.float64)


async def _summarize_cluster_titles(
    ordered_titles: Sequence[str],
    model: str,
    sem: asyncio.Semaphore,
    api_key: str | None = None,
) -> str:
    extra: dict[str, Any] = {"api_key": api_key.strip()} if (api_key and api_key.strip()) else {}
    sampled = list(ordered_titles[:MAX_TITLES_FOR_SUMMARY])
    titles_block = "\n".join(f"- {t}" for t in sampled)
    messages = [
        {
            "role": "user",
            "content": f"{VERBATIM_CLUSTER_SUMMARY_PROMPT}\n\nPages in this group:\n{titles_block}",
        }
    ]
    async with sem:
        resp = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=0.0,
            **extra,
        )
    raw = _extract_completion_text(resp)
    return _cap_thirty_words(raw)


async def build_page_clusters(
    db: AsyncSession,
    *,
    namespace_id: UUID,
    settings: Settings,
    num_clusters: int | None = None,
    force: bool = False,
    api_key: str | None = None,
) -> list[TopicCluster]:
    """Build offline TopicCluster nodes over uncategorized READY pages (Spec §5.2)."""
    stmt = (
        select(DocumentPage)
        .where(
            DocumentPage.namespace_id == namespace_id,
            DocumentPage.status == "READY",
            DocumentPage.category_id.is_(None),
        )
        .order_by(DocumentPage.doc_order.asc())
    )
    pages = list((await db.execute(stmt)).scalars().all())
    if not pages:
        raise ValueError("No uncategorized READY pages found in this namespace.")

    if len(pages) <= 650 and not force and num_clusters is None:
        raise ValueError(
            f"Namespace has {len(pages)} uncategorized pages (<= 650). "
            "Pass force=True or specify num_clusters explicitly to cluster a small corpus."
        )

    target_k = num_clusters if num_clusters is not None else max(1, round(len(pages) / 100))
    target_k = max(1, min(target_k, len(pages)))

    opening_texts = [
        f"{p.title}. {p.first_sentence}".strip() if p.first_sentence else p.title
        for p in pages
    ]
    vectors = await _embed_page_openings(opening_texts, settings.CLUSTER_EMBEDDING_MODEL, api_key=api_key)
    x_norm = _l2_normalize(vectors)
    labels, centroids = cosine_kmeans(x_norm, target_k)

    for p in pages:
        p.cluster_id = None
        p.cluster_rank = None
    await db.execute(delete(TopicCluster).where(TopicCluster.namespace_id == namespace_id))
    await db.flush()

    cluster_groups: list[tuple[int, list[DocumentPage]]] = []
    for c_idx in range(target_k):
        member_indices = [i for i, lbl in enumerate(labels.tolist()) if lbl == c_idx]
        if not member_indices:
            continue
        centroid = centroids[c_idx]
        scored_members = sorted(
            member_indices,
            key=lambda i: (-float(x_norm[i] @ centroid), pages[i].doc_order),
        )
        ordered_pages = [pages[i] for i in scored_members]
        cluster_groups.append((len(cluster_groups), ordered_pages))

    sem = asyncio.Semaphore(settings.DECISION_CONCURRENCY)
    summaries = await asyncio.gather(
        *(
            _summarize_cluster_titles(
                [p.title for p in grp_pages],
                settings.CLUSTER_SUMMARY_MODEL,
                sem,
                api_key=api_key,
            )
            for _, grp_pages in cluster_groups
        )
    )
    built_clusters: list[TopicCluster] = []
    for (cluster_index, grp_pages), summary in zip(cluster_groups, summaries, strict=True):
        tc = TopicCluster(
            namespace_id=namespace_id,
            cluster_index=cluster_index,
            summary=summary,
            page_count=len(grp_pages),
        )
        db.add(tc)
        await db.flush()

        for rank, page in enumerate(grp_pages):
            page.cluster_id = tc.id
            page.cluster_rank = rank

        built_clusters.append(tc)

    await db.commit()
    for tc in built_clusters:
        await db.refresh(tc)
    return built_clusters