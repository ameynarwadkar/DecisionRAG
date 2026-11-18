from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.models import DocumentPage, Namespace
from dec_rag.services.ingestion import ingest_document

DEMO_NAMESPACE_NAME = "Cloud Platform Docs"
DEMO_NAMESPACE_SLUG = "cloud-platform-docs"

DEMO_DOCUMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "Security",
        "eu-data-residency.md",
        """# EU Data Residency & Zero-Retention

## Standard EU Routing

Workspaces pinned to `eu-central-1` process synchronous inference in Frankfurt by default. During regional capacity brownouts, traffic automatically fails over to `us-east-1` to preserve availability unless strict residency lock is enabled.

## Strict Residency Lock

Enabling `strict_eu_lock=true` disables cross-Atlantic failover completely. If both `eu-central-1` (Frankfurt) and `eu-west-1` (Dublin) reach saturation or suffer an outage, requests immediately return HTTP `503 RegionalCapacityExhausted` instead of routing outside the EEA.

## Zero-Retention Memory Policy

With zero-retention active, prompt and completion buffers exist only in GPU SRAM and host memory for the duration of the HTTP stream and are zeroed before socket teardown. No customer payloads are written to NVMe or object storage.

## Telemetry & Billing Replication

Even when `strict_eu_lock=true` is active, non-content billing counters (workspace ID, timestamp, and integer token counts, with zero prompt text) replicate to the global billing ledger in `us-east-1` every 60 seconds.
""",
    ),
    (
        "Security",
        "soc2-audit-logs.md",
        """# SOC 2 Audit Logs & Customer-Managed Keys

## Immutable Audit Stream

Every administrative action, API key rotation, and residency flag change emits a signed JSON audit event within 5 seconds. Audit logs are retained for 400 days and can stream to S3 or Datadog via webhook.

## Customer-Managed Encryption Keys (CMEK)

Enterprise workspaces can register an AWS KMS or Azure Key Vault key ARN to wrap workspace metadata. Revoking the KMS grant invalidates cached envelope keys across all serving pods within 5 minutes.

## Emergency Key Rotation

Rotating a compromised API key supports a configurable grace window between 0 and 3600 seconds so production workers can roll secrets without dropping in-flight requests.
""",
    ),
    (
        "Rate Limits",
        "rate-limits-and-bursts.md",
        """# Token Rate Limits & Burst Quotas

## Tiered Throughput Quotas

Growth workspaces receive 400,000 input tokens per minute (TPM) and 1,200 requests per minute (RPM) enforced via a sliding token-bucket limiter replenished every 100 milliseconds.

## Short Burst Allowance

Workspaces with unused capacity over the preceding 60 seconds can burst up to 2.5x their baseline RPM for up to 8 seconds before HTTP `429 RateLimitExceeded` responses begin.

## Cached Context Discount

Shared-prefix tokens matched in the cluster KV cache count at 0.2x weight against the workspace TPM quota and carry a 75% per-token billing discount.
""",
    ),
    (
        "Rate Limits",
        "batch-async-queues.md",
        """# Async Batch Jobs & Webhook Retries

## 24-Hour Batch Queue

JSONL batch files up to 50,000 requests execute outside real-time RPM quotas at a 50% discount, completing within 24 hours.

## Webhook Delivery & Backoff

Completion webhooks sign payloads with HMAC-SHA256 (`X-Signature-256`) and retry non-2xx responses up to 6 times using exponential backoff (5s, 30s, 2m, 15m, 1h, 4h).
""",
    ),
    (
        "SLA",
        "uptime-sla-credits.md",
        """# Uptime SLA & Service Credits

## Monthly Availability Commitment

Enterprise deployments carry a 99.95% monthly uptime SLA measured across 1-minute synthetic probes. Client-side 4xx errors and intentional `503 RegionalCapacityExhausted` responses triggered by `strict_eu_lock=true` are excluded from downtime calculations.

## Service Credit Schedule

Monthly availability between 99.0% and 99.95% earns a 10% invoice credit; availability between 95.0% and 99.0% earns a 25% credit; availability below 95.0% earns a 50% credit. Credit claims must be filed within 30 days of month close.
""",
    ),
    (
        "SLA",
        "incident-severity-tiers.md",
        """# Incident Severity Tiers & Response Targets

## Sev-1 Critical Outage

Complete regional unavailability or elevated 5xx rates above 5% trigger automated pager escalation with a 15-minute engineering response target and status-page updates every 30 minutes.

## Sev-2 Degraded Latency

P99 time-to-first-token exceeding 3x baseline for more than 10 consecutive minutes triggers a Sev-2 investigation with a 1-hour response target.
""",
    ),
)


async def ensure_demo_corpus(db: AsyncSession) -> Namespace:
    """Ensure the built-in Cloud Platform Docs demo corpus exists and has pages."""
    ns = (
        await db.execute(
            select(Namespace).where(Namespace.slug == DEMO_NAMESPACE_SLUG)
        )
    ).scalar_one_or_none()

    if ns is not None:
        existing_pages = int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(DocumentPage)
                    .where(DocumentPage.namespace_id == ns.id)
                )
            ).scalar_one()
        )
        if existing_pages > 0:
            return ns
    else:
        ns = Namespace(
            name=DEMO_NAMESPACE_NAME,
            slug=DEMO_NAMESPACE_SLUG,
            description="Built-in cloud infrastructure corpus (Security, Rate Limits, SLA)",
            default_profile="auto",
        )
        db.add(ns)
        await db.commit()
        await db.refresh(ns)

    for category, filename, content in DEMO_DOCUMENTS:
        await ingest_document(
            db,
            namespace_id=ns.id,
            filename=filename,
            raw_bytes=content.encode("utf-8"),
            content_type="text/markdown",
            category_name=category,
        )

    await db.refresh(ns)
    return ns
