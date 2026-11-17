from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.auth import AuthContext, require_api_key
from dec_rag.config import Settings, get_settings
from dec_rag.database import get_db
from dec_rag.models import CategoryNode, DocumentPage, Namespace, TopicCluster
from dec_rag.schemas import (
    CategoryRead,
    ClusterBuildRequest,
    ClusterBuildResponse,
    ClusterRead,
    NamespaceCreate,
    NamespaceDetailRead,
    NamespaceRead,
)
from dec_rag.services.clustering import build_page_clusters
from dec_rag.services.ingestion import slugify
from dec_rag.schemas import apply_runtime_credentials

router = APIRouter(prefix="/namespaces", tags=["namespaces"])


async def _populate_namespace_counts(db: AsyncSession, ns: Namespace) -> NamespaceRead:
    page_count = int(
        (
            await db.execute(
                select(func.count()).select_from(DocumentPage).where(DocumentPage.namespace_id == ns.id)
            )
        ).scalar_one()
    )
    cat_count = int(
        (
            await db.execute(
                select(func.count()).select_from(CategoryNode).where(CategoryNode.namespace_id == ns.id)
            )
        ).scalar_one()
    )
    cluster_count = int(
        (
            await db.execute(
                select(func.count()).select_from(TopicCluster).where(TopicCluster.namespace_id == ns.id)
            )
        ).scalar_one()
    )
    return NamespaceRead(
        id=ns.id,
        name=ns.name,
        slug=ns.slug,
        description=ns.description,
        system_prompt=ns.system_prompt,
        default_profile=ns.default_profile,  # type: ignore[arg-type]
        created_at=ns.created_at,
        page_count=page_count,
        category_count=cat_count,
        cluster_count=cluster_count,
    )


@router.post("", response_model=NamespaceRead, status_code=status.HTTP_201_CREATED)
async def create_namespace(
    body: NamespaceCreate,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> NamespaceRead:
    if auth.namespace_id is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Namespace-scoped API keys cannot create new namespaces.",
        )

    target_slug = slugify(body.slug or body.name)
    existing = (
        await db.execute(select(Namespace).where(Namespace.slug == target_slug))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Namespace with slug '{target_slug}' already exists.",
        )

    ns = Namespace(
        name=body.name.strip(),
        slug=target_slug,
        description=body.description,
        system_prompt=body.system_prompt,
        default_profile=body.default_profile,
    )
    db.add(ns)
    await db.commit()
    await db.refresh(ns)
    return await _populate_namespace_counts(db, ns)


@router.get("", response_model=list[NamespaceRead])
async def list_namespaces(
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> list[NamespaceRead]:
    stmt = select(Namespace).order_by(Namespace.created_at.desc())
    if auth.namespace_id is not None:
        stmt = stmt.where(Namespace.id == auth.namespace_id)
    namespaces = list((await db.execute(stmt)).scalars().all())
    return [await _populate_namespace_counts(db, ns) for ns in namespaces]


@router.get("/{ns_id}", response_model=NamespaceDetailRead)
async def get_namespace(
    ns_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> NamespaceDetailRead:
    auth.ensure_namespace_access(ns_id)
    ns = (await db.execute(select(Namespace).where(Namespace.id == ns_id))).scalar_one_or_none()
    if ns is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")

    base = await _populate_namespace_counts(db, ns)
    cats = list(
        (
            await db.execute(
                select(CategoryNode)
                .where(CategoryNode.namespace_id == ns.id)
                .order_by(CategoryNode.order_index.asc())
            )
        ).scalars().all()
    )
    clusters = list(
        (
            await db.execute(
                select(TopicCluster)
                .where(TopicCluster.namespace_id == ns.id)
                .order_by(TopicCluster.cluster_index.asc())
            )
        ).scalars().all()
    )
    return NamespaceDetailRead(
        **base.model_dump(),
        categories=[CategoryRead.model_validate(c) for c in cats],
        clusters=[ClusterRead.model_validate(tc) for tc in clusters],
    )


@router.delete("/{ns_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_namespace(
    ns_id: UUID,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
) -> Response:
    auth.ensure_namespace_access(ns_id)
    ns = (await db.execute(select(Namespace).where(Namespace.id == ns_id))).scalar_one_or_none()
    if ns is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")
    await db.delete(ns)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/{ns_id}/clusters/build", response_model=ClusterBuildResponse)
async def trigger_cluster_build(
    ns_id: UUID,
    body: ClusterBuildRequest,
    auth: AuthContext = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ClusterBuildResponse:
    auth.ensure_namespace_access(ns_id)
    ns = (await db.execute(select(Namespace).where(Namespace.id == ns_id))).scalar_one_or_none()
    if ns is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Namespace not found.")

    effective_settings = apply_runtime_credentials(settings, body.runtime_credentials)
    gen_key = body.runtime_credentials.generator_api_key if body.runtime_credentials else None

    try:
        built = await build_page_clusters(
            db,
            namespace_id=ns.id,
            settings=effective_settings,
            num_clusters=body.num_clusters,
            force=body.force,
            api_key=gen_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    pages_clustered = sum(tc.page_count for tc in built)
    return ClusterBuildResponse(
        namespace_id=ns.id,
        clusters_built=len(built),
        pages_clustered=pages_clustered,
        clusters=[ClusterRead.model_validate(tc) for tc in built],
    )