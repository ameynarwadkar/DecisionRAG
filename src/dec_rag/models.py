from datetime import UTC, datetime
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dec_rag.database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Namespace(Base):
    __tablename__ = "namespaces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_profile: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    categories: Mapped[list["CategoryNode"]] = relationship(
        back_populates="namespace",
        cascade="all, delete-orphan",
    )
    clusters: Mapped[list["TopicCluster"]] = relationship(
        back_populates="namespace",
        cascade="all, delete-orphan",
    )
    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="namespace",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["ConversationSession"]] = relationship(
        back_populates="namespace",
        cascade="all, delete-orphan",
    )


class CategoryNode(Base):
    __tablename__ = "category_nodes"
    __table_args__ = (UniqueConstraint("namespace_id", "slug", name="uq_category_ns_slug"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_manual_description: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    namespace: Mapped["Namespace"] = relationship(back_populates="categories")
    pages: Mapped[list["DocumentPage"]] = relationship(back_populates="category")


class TopicCluster(Base):
    __tablename__ = "topic_clusters"
    __table_args__ = (UniqueConstraint("namespace_id", "cluster_index", name="uq_cluster_ns_idx"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cluster_index: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    namespace: Mapped["Namespace"] = relationship(back_populates="clusters")
    pages: Mapped[list["DocumentPage"]] = relationship(back_populates="cluster")


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (Index("ix_pages_ns_order", "namespace_id", "doc_order"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("category_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    cluster_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("topic_clusters.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    cluster_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doc_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    first_sentence: Mapped[str] = mapped_column(Text, default="", nullable=False)
    headings_summary: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), default="text/plain", nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    passage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    namespace: Mapped["Namespace"] = relationship(back_populates="pages")
    category: Mapped["CategoryNode | None"] = relationship(back_populates="pages")
    cluster: Mapped["TopicCluster | None"] = relationship(back_populates="pages")
    headings: Mapped[list["SectionHeading"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
    )
    passages: Mapped[list["Passage"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
    )


class SectionHeading(Base):
    __tablename__ = "section_headings"
    __table_args__ = (Index("ix_headings_page_order", "page_id", "heading_index"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    heading_index: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    heading_path: Mapped[str] = mapped_column(Text, nullable=False)

    page: Mapped["DocumentPage"] = relationship(back_populates="headings")


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (Index("ix_passages_page_order", "page_id", "passage_index"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    heading_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("section_headings.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    passage_index: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str] = mapped_column(Text, default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    norm_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    page: Mapped["DocumentPage"] = relationship(back_populates="passages")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turns_json: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list, nullable=False)
    notepad_page_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    namespace: Mapped["Namespace"] = relationship(back_populates="sessions")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    namespace_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("namespaces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)