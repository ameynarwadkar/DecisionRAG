import re
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.models import CategoryNode, DocumentPage, Passage, SectionHeading
from dec_rag.services.parser import parse_document, strip_part_suffix

SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = SLUG_CLEAN_RE.sub("-", value.lower()).strip("-")
    return slug or "item"


async def refresh_category_auto_description(
    db: AsyncSession,
    category: CategoryNode,
) -> None:
    """Refresh CategoryNode auto-description per Spec §5.1 / Appendix A.3."""
    if category.is_manual_description:
        return

    stmt = (
        select(DocumentPage)
        .where(
            DocumentPage.category_id == category.id,
            DocumentPage.status == "READY",
        )
        .order_by(DocumentPage.doc_order.asc())
    )
    pages = list((await db.execute(stmt)).scalars().all())
    if not pages:
        return

    landing = pages[0]
    sample_titles = "; ".join(p.title for p in pages[:3])
    category.description = f"{category.name}: {landing.first_sentence} (Pages include: {sample_titles})"


async def _get_or_create_category(
    db: AsyncSession,
    *,
    namespace_id: UUID,
    category_name: str,
    category_description: str | None,
) -> CategoryNode:
    clean_name = category_name.strip()
    cat_slug = slugify(clean_name)
    stmt = select(CategoryNode).where(
        CategoryNode.namespace_id == namespace_id,
        CategoryNode.slug == cat_slug,
    )
    category = (await db.execute(stmt)).scalar_one_or_none()
    manual_desc = category_description.strip() if category_description and category_description.strip() else None

    if category is None:
        max_order_stmt = select(func.coalesce(func.max(CategoryNode.order_index), -1)).where(
            CategoryNode.namespace_id == namespace_id
        )
        next_order = int((await db.execute(max_order_stmt)).scalar_one()) + 1
        category = CategoryNode(
            namespace_id=namespace_id,
            name=clean_name,
            slug=cat_slug,
            description=manual_desc,
            is_manual_description=manual_desc is not None,
            order_index=next_order,
        )
        db.add(category)
        await db.flush()
    elif manual_desc is not None:
        category.description = manual_desc
        category.is_manual_description = True
        await db.flush()

    return category


async def ingest_document(
    db: AsyncSession,
    *,
    namespace_id: UUID,
    filename: str,
    raw_bytes: bytes,
    content_type: str | None = None,
    category_name: str | None = None,
    category_description: str | None = None,
    merge_part_suffix: bool = False,
    max_passage_chars: int = 1200,
) -> tuple[DocumentPage, bool]:
    """Parse and persist a structured DocumentPage. Returns (page, merged_into_existing)."""
    parsed = parse_document(
        raw_bytes=raw_bytes,
        filename=filename,
        content_type=content_type,
        max_passage_chars=max_passage_chars,
    )

    category: CategoryNode | None = None
    if category_name and category_name.strip():
        category = await _get_or_create_category(
            db,
            namespace_id=namespace_id,
            category_name=category_name,
            category_description=category_description,
        )

    effective_title = strip_part_suffix(parsed.title) if merge_part_suffix else parsed.title

    if merge_part_suffix:
        existing_stmt = (
            select(DocumentPage)
            .where(
                DocumentPage.namespace_id == namespace_id,
                DocumentPage.status == "READY",
                DocumentPage.category_id == (category.id if category else None),
            )
            .order_by(DocumentPage.doc_order.asc())
        )
        candidates = list((await db.execute(existing_stmt)).scalars().all())
        target = next(
            (
                p
                for p in candidates
                if strip_part_suffix(p.title).lower() == effective_title.lower()
            ),
            None,
        )
        if target is not None:
            max_h_stmt = select(func.coalesce(func.max(SectionHeading.heading_index), -1)).where(
                SectionHeading.page_id == target.id
            )
            h_offset = int((await db.execute(max_h_stmt)).scalar_one()) + 1
            p_offset = target.passage_count

            heading_map: dict[int, SectionHeading] = {}
            for ph in parsed.headings:
                sh = SectionHeading(
                    page_id=target.id,
                    namespace_id=namespace_id,
                    heading_index=h_offset + ph.heading_index,
                    level=ph.level,
                    title=ph.title,
                    heading_path=ph.heading_path,
                )
                db.add(sh)
                heading_map[ph.heading_index] = sh
            await db.flush()

            for pp in parsed.passages:
                linked_h = heading_map.get(pp.heading_index) if pp.heading_index is not None else None
                db.add(
                    Passage(
                        page_id=target.id,
                        namespace_id=namespace_id,
                        heading_id=linked_h.id if linked_h else None,
                        passage_index=p_offset + pp.passage_index,
                        heading_path=pp.heading_path,
                        content=pp.content,
                        norm_hash=pp.norm_hash,
                        token_estimate=pp.token_estimate,
                    )
                )

            merged_headings = list(target.headings_summary)
            for h_title in parsed.headings_summary:
                if h_title not in merged_headings:
                    merged_headings.append(h_title)

            target.title = effective_title
            target.headings_summary = merged_headings
            target.passage_count = p_offset + len(parsed.passages)
            target.file_size = target.file_size + len(raw_bytes)
            await db.flush()

            if category is not None:
                await refresh_category_auto_description(db, category)

            await db.commit()
            await db.refresh(target)
            return target, True

    max_doc_order_stmt = select(func.coalesce(func.max(DocumentPage.doc_order), -1)).where(
        DocumentPage.namespace_id == namespace_id
    )
    next_doc_order = int((await db.execute(max_doc_order_stmt)).scalar_one()) + 1

    page = DocumentPage(
        namespace_id=namespace_id,
        category_id=category.id if category else None,
        doc_order=next_doc_order,
        filename=filename,
        title=effective_title,
        first_sentence=parsed.first_sentence,
        headings_summary=parsed.headings_summary,
        content_type=content_type or "text/plain",
        file_size=len(raw_bytes),
        passage_count=len(parsed.passages),
        status="READY",
    )
    db.add(page)
    await db.flush()

    heading_map = {}
    for ph in parsed.headings:
        sh = SectionHeading(
            page_id=page.id,
            namespace_id=namespace_id,
            heading_index=ph.heading_index,
            level=ph.level,
            title=ph.title,
            heading_path=ph.heading_path,
        )
        db.add(sh)
        heading_map[ph.heading_index] = sh
    await db.flush()

    for pp in parsed.passages:
        linked_h = heading_map.get(pp.heading_index) if pp.heading_index is not None else None
        db.add(
            Passage(
                page_id=page.id,
                namespace_id=namespace_id,
                heading_id=linked_h.id if linked_h else None,
                passage_index=pp.passage_index,
                heading_path=pp.heading_path,
                content=pp.content,
                norm_hash=pp.norm_hash,
                token_estimate=pp.token_estimate,
            )
        )
    await db.flush()

    if category is not None:
        await refresh_category_auto_description(db, category)

    await db.commit()
    await db.refresh(page)
    return page, False