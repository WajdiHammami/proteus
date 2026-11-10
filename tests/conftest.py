import io

import pytest
from pypdf import PdfReader, PdfWriter

from src.adapters.models.source import Source, SourceType


# Page size (A4 in points) used for full-page "scanned" images.
_W, _H = 595, 842


def _escape_pdf_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[dict], title: str | None = None) -> bytes:
    """
    Builds a minimal, valid PDF without external dependencies.

    Each page spec may contain:
      text:       str  -> drawn with Helvetica (real text layer)
      image:      bool -> full-page image XObject (looks like a scan)
      form_image: bool -> image nested inside a form XObject
      lines:      list[(x_offset_pt, str)] -> one line each, 14pt apart, placed by position
    An empty dict gives a blank page.
    """
    objects: list[bytes] = []  # objects[i] is object number i + 1

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    def stream(dictionary: str, data: bytes) -> bytes:
        return f"<< {dictionary} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"

    catalog = add(b"")  # placeholder, filled once the pages tree exists
    pages_root = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    for spec in pages:
        ops: list[str] = []
        xobjects: dict[str, int] = {}

        if spec.get("image") or spec.get("form_image"):
            image = add(stream(
                "/Type /XObject /Subtype /Image /Width 1 /Height 1 "
                "/ColorSpace /DeviceGray /BitsPerComponent 8",
                b"\x80",
            ))
            if spec.get("form_image"):
                form = add(stream(
                    f"/Type /XObject /Subtype /Form /BBox [0 0 {_W} {_H}] "
                    f"/Resources << /XObject << /Im1 {image} 0 R >> >>",
                    f"q {_W} 0 0 {_H} 0 0 cm /Im1 Do Q".encode(),
                ))
                xobjects["Fm1"] = form
                ops.append("/Fm1 Do")
            else:
                xobjects["Im1"] = image
                ops.append(f"q {_W} 0 0 {_H} 0 0 cm /Im1 Do Q")

        if spec.get("text"):
            ops.append(f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_string(spec['text'])}) Tj ET")

        # Positioned lines: indentation comes from the x offset, not from space characters,
        # which is how most real PDFs lay out indented code.
        y = 700
        for x_offset, line in spec.get("lines", []):
            ops.append(f"BT /F1 10 Tf {72 + x_offset} {y} Td ({_escape_pdf_string(line)}) Tj ET")
            y -= 14

        content = add(stream("", "\n".join(ops).encode()))
        resources = f"/Font << /F1 {font} 0 R >>"
        if xobjects:
            refs = " ".join(f"/{name} {num} 0 R" for name, num in xobjects.items())
            resources += f" /XObject << {refs} >>"
        page_ids.append(add(
            f"<< /Type /Page /Parent {pages_root} 0 R /MediaBox [0 0 {_W} {_H}] "
            f"/Resources << {resources} >> /Contents {content} 0 R >>".encode()
        ))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_root - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_root} 0 R >>".encode()
    info = add(f"<< /Title ({_escape_pdf_string(title)}) >>".encode()) if title else None

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_at = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    trailer = f"/Size {len(objects) + 1} /Root {catalog} 0 R"
    if info:
        trailer += f" /Info {info} 0 R"
    out.write(f"trailer\n<< {trailer} >>\nstartxref\n{xref_at}\n%%EOF\n".encode())
    return out.getvalue()


def encrypt_pdf(pdf_bytes: bytes, user_password: str, owner_password: str = "owner") -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_bytes)))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


LONG_TEXT = "This page has a proper text layer with plenty of characters."


@pytest.fixture
def write_pdf(tmp_path):
    """Writes a PDF built from page specs to tmp_path and returns its path as str."""
    counter = iter(range(1_000_000))

    def _write(pages: list[dict], title: str | None = None, raw: bytes | None = None) -> str:
        path = tmp_path / f"doc_{next(counter)}.pdf"
        path.write_bytes(raw if raw is not None else build_pdf(pages, title=title))
        return str(path)

    return _write


@pytest.fixture
def make_source():
    def _make(url: str, type: SourceType, id: str = "src-1") -> Source:
        return Source(id=id, url=url, type=type)

    return _make
