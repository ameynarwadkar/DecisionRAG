import io

import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dec_rag.models import CategoryNode, Namespace, Passage
from dec_rag.services.ingestion import ingest_document
from dec_rag.services.parser import parse_document


def _make_simple_pdf(heading: str, sentence: str) -> bytes:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    content = f"BT /F1 12 Tf 72 720 Td ({heading}) Tj 0 -36 Td ({sentence}) Tj ET".encode("ascii")
    stream.set_data(content)
    page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_html_dom_reading_order_and_headings() -> None:
    html = b"""
    <html>
      <head>
        <title>Account Billing Guide</title>
        <script>console.log("ignored");</script>
      </head>
      <body>
        <h1>Subscriptions</h1>
        <p>Annual plans renew automatically every 12 months.</p>
        <h2>Pricing Tiers</h2>
        <ul>
          <li>Starter tier costs 10 USD per month.</li>
          <li>Pro tier costs 29 USD per month.</li>
        </ul>
        <table>
          <tr><th>Region</th><th>Tax</th></tr>
          <tr><td>EU</td><td>VAT applied</td></tr>
        </table>
      </body>
    </html>
    """
    doc = parse_document(html, "billing.html", "text/html")
    assert doc.title == "Account Billing Guide"
    assert doc.first_sentence == "Annual plans renew automatically every 12 months."
    assert [p.passage_index for p in doc.passages] == [0, 1, 2, 3, 4]
    assert doc.passages[0].heading_path == "Subscriptions"
    assert doc.passages[1].heading_path == "Subscriptions > Pricing Tiers"
    assert doc.passages[3].content == "Region | Tax"
    assert doc.passages[4].content == "EU | VAT applied"
    assert all("ignored" not in p.content for p in doc.passages)


def test_markdown_nested_headings_and_sentence_splitting() -> None:
    long_para = " ".join(
        [f"Sentence number {i} explains the household income threshold in detail." for i in range(12)]
    )
    md = f"""# Housing Assistance Guide

## Eligibility

### Income limits

{long_para}
""".encode("utf-8")

    doc = parse_document(md, "housing.md", "text/markdown", max_passage_chars=260)
    assert doc.title == "Housing Assistance Guide"
    assert doc.headings_summary == ["Eligibility"]
    assert len(doc.passages) >= 3
    for idx, p in enumerate(doc.passages):
        assert p.passage_index == idx
        assert p.heading_path == "Eligibility > Income limits"
        assert len(p.content) <= 260


def test_plain_text_and_pdf_layout_heading_recovery() -> None:
    txt = b"""Benefit Appeals Guide

1. Filing Deadlines

Applicants must submit written appeals within 30 calendar days of denial.

1.1 Late Submissions

Good cause extensions require supporting medical documentation.
"""
    txt_doc = parse_document(txt, "appeals.txt", "text/plain")
    assert txt_doc.title == "Benefit Appeals Guide"
    assert txt_doc.passages[0].heading_path == "1. Filing Deadlines"
    assert txt_doc.passages[1].heading_path == "1. Filing Deadlines > 1.1 Late Submissions"

    pdf_bytes = _make_simple_pdf("Appeals Process", "Submit appeals within 14 days.")
    pdf_doc = parse_document(pdf_bytes, "appeals_form.pdf", "application/pdf")
    assert pdf_doc.title == "appeals form"
    assert len(pdf_doc.passages) == 1
    assert pdf_doc.passages[0].heading_path == "Appeals Process"
    assert pdf_doc.passages[0].content == "Submit appeals within 14 days."


async def test_ingestion_category_auto_description_and_part_merging(db_session: AsyncSession) -> None:
    ns = Namespace(name="Help Center", slug="help-center")
    db_session.add(ns)
    await db_session.commit()

    p1, merged1 = await ingest_document(
        db_session,
        namespace_id=ns.id,
        filename="shipping_part1.md",
        raw_bytes=b"# Shipping Policy #1\n\n## Domestic\n\nDomestic orders arrive in 3 business days.",
        category_name="Orders & Shipping",
        merge_part_suffix=True,
    )
    assert merged1 is False
    assert p1.title == "Shipping Policy"
    assert p1.passage_count == 1

    p2, merged2 = await ingest_document(
        db_session,
        namespace_id=ns.id,
        filename="shipping_part2.md",
        raw_bytes=b"# Shipping Policy #2\n\n## International\n\nInternational orders arrive in 10 business days.",
        category_name="Orders & Shipping",
        merge_part_suffix=True,
    )
    assert merged2 is True
    assert p2.id == p1.id
    assert p2.passage_count == 2
    assert p2.headings_summary == ["Domestic", "International"]

    # Add a second distinct page under the same category to verify auto-description refresh
    p3, merged3 = await ingest_document(
        db_session,
        namespace_id=ns.id,
        filename="returns.md",
        raw_bytes=b"# Return Labels\n\nPrint your prepaid label from the portal.",
        category_name="Orders & Shipping",
    )
    assert merged3 is False
    assert p3.doc_order == 1

    cat = (
        await db_session.execute(select(CategoryNode).where(CategoryNode.namespace_id == ns.id))
    ).scalar_one()
    assert cat.description == (
        "Orders & Shipping: Domestic orders arrive in 3 business days. "
        "(Pages include: Shipping Policy; Return Labels)"
    )

    passages = list(
        (
            await db_session.execute(
                select(Passage).where(Passage.page_id == p1.id).order_by(Passage.passage_index.asc())
            )
        ).scalars().all()
    )
    assert [p.passage_index for p in passages] == [0, 1]
    assert [p.heading_path for p in passages] == ["Domestic", "International"]