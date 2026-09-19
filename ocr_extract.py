"""
Document OCR + field extraction.

Handles two source document types:
  - RC book (registration certificate): gives registration number,
    registration date (-> vehicle age), vehicle class/type, seats, GVW.
  - Previous insurance policy/certificate: gives IDV (Sum Insured),
    previous NCB%, policy dates, vehicle details as a fallback.

Because free OCR (Tesseract) will misread some characters on real-world
scans, this module is deliberately permissive: it returns whatever it can
find plus a `confidence` note per field, and the client UI lets the user
correct any field before calculation runs. Do not treat this output as
final — it's a first draft for the human to check.
"""

import io
import re
from datetime import date

import pytesseract
from PIL import Image
import pymupdf as fitz  # renders PDF pages without needing a system binary


def ocr_text_from_file(file_bytes: bytes, content_type: str = None) -> str:
    """Run OCR on an image or PDF and return raw extracted text.
    Detects PDFs by their file signature (%PDF magic bytes) rather than
    trusting the browser-supplied content_type, which can come back empty
    or generic on some browsers/devices and would otherwise cause a real
    PDF to be mishandled as an image."""
    is_pdf = file_bytes[:5] == b"%PDF-" or content_type == "application/pdf"

    if is_pdf:
        text_parts = []
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            # 300 DPI equivalent: default page is 72 DPI, so scale ~4.17x
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text_parts.append(pytesseract.image_to_string(image))
        pdf.close()
        return "\n".join(text_parts)

    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


def detect_doc_type(text: str) -> str:
    """Return 'rc', 'policy', or 'unknown' based on keyword heuristics."""
    lower = text.lower()
    rc_hits = sum(k in lower for k in ["registration certificate", "chassis no", "engine no", "rc book", "form 23"])
    policy_hits = sum(k in lower for k in ["policy no", "idv", "sum insured", "ncb", "certificate of insurance"])
    if rc_hits > policy_hits:
        return "rc"
    if policy_hits > rc_hits:
        return "policy"
    return "unknown"


DATE_RE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
REG_NO_RE = re.compile(r"\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{0,3}\s?\d{3,4})\b")
IDV_RE = re.compile(r"(?:idv|sum insured)[^\d₹]{0,15}([\d,]+)", re.IGNORECASE)
NCB_RE = re.compile(r"ncb[^\d]{0,10}(\d{1,3})\s*%", re.IGNORECASE)
# "Seating capacity" as one phrase, OR "Seating" followed eventually by a
# number even when other column headers (Standing/Sleeper Capacity etc.)
# sit in between, as on RC smart cards — a wide but bounded window avoids
# matching some unrelated number much further down the page.
SEATS_RE = re.compile(r"(?:seating[^\d]{0,60}capacity|no\.? of seats|seats)[^\d]{0,60}(\d{1,3})", re.IGNORECASE)
GVW_RE = re.compile(r"(?:gvw|gross vehicle weight)[^\d]{0,10}([\d,]+)", re.IGNORECASE)
CC_RE = re.compile(r"(?:cubic capacity|engine capacity|cc)[:\s/]{0,40}(\d{2,5})(?:\.\d+)?\s*(?:cc)?", re.IGNORECASE)
# Matches "Engine No", "Engine Number", "Engine/Motor Number", "Motor No" —
# real RC smart cards commonly use the full word "Number" and combine
# Engine/Motor into one label, not just the abbreviated "No." form.
ENGINE_NO_RE = re.compile(r"(?:engine\s*/?\s*motor|engine|motor)\s*(?:no\.?|number)[:\s]*([A-Z0-9]{5,20})", re.IGNORECASE)
CHASSIS_NO_RE = re.compile(r"chassis\s*(?:no\.?|number)[:\s]*([A-Z0-9]{5,20})", re.IGNORECASE)
# Owner/customer name — heuristic only; RC/policy layouts vary a lot, so
# this is a first guess for the user to confirm or fix on the review screen.
# Deliberately does NOT include a bare "name" fallback — RC cards have
# several "...Name" labels (Maker's Name, Model Name) that would false-match.
NAME_RE = re.compile(r"(?:owner'?s?\s*name|name\s*of\s*owner|insured\s*name)[:\s]*([A-Z][A-Za-z .]{2,40})", re.IGNORECASE)

# Vehicle type keywords, checked in order — first match wins. These are
# common terms found on RC books' "Class of Vehicle" / "Vehicle Class"
# field and on policy schedules. Tune against real samples over time.
VEHICLE_TYPE_KEYWORDS = [
    ("Two Wheelers", ["motor cycle", "m-cycle", "scooter", "moped", "two wheeler"]),
    ("Taxi", ["taxi", "motor cab", "maxi cab"]),
    ("PCV > 6 PSGR", ["omnibus", "pcv", "passenger carrying vehicle"]),
    ("GCV 4 Wheelers", ["goods carriage", "gcv", "lorry", "truck"]),
    ("Private Cars", ["motor car", "m-car", "private car", "saloon", "hatchback", "suv"]),
]


def detect_vehicle_type(text: str) -> str | None:
    """Best-effort vehicle class from RC/policy keywords. Returns None if
    nothing matches — the client UI's dropdown default then applies, and
    the user should confirm/correct this either way."""
    lower = text.lower()
    for vehicle_type, keywords in VEHICLE_TYPE_KEYWORDS:
        if any(k in lower for k in keywords):
            return vehicle_type
    return None


def _parse_date(match) -> date | None:
    try:
        d, m, y = match.groups()
        y = int(y)
        if y < 100:
            y += 2000
        return date(y, int(m), int(d))
    except Exception:
        return None


def vehicle_age_years(reg_date: date | None) -> int | None:
    if not reg_date:
        return None
    today = date.today()
    years = today.year - reg_date.year - ((today.month, today.day) < (reg_date.month, reg_date.day))
    return max(years, 0)


def extract_fields(text: str, doc_type: str) -> dict:
    """Best-effort field extraction. Every value should be treated as a
    draft for the user to confirm/correct in the UI, not ground truth."""
    fields = {}

    reg_no_match = REG_NO_RE.search(text)
    if reg_no_match:
        fields["registration_number"] = reg_no_match.group(1).replace(" ", "")

    dates_found = [_parse_date(m) for m in DATE_RE.finditer(text)]
    dates_found = [d for d in dates_found if d]
    if dates_found:
        # Assume the earliest plausible date is the registration date
        reg_date = min(dates_found)
        fields["registration_date"] = reg_date.isoformat()
        fields["vehicle_age_years"] = vehicle_age_years(reg_date)

    idv_match = IDV_RE.search(text)
    if idv_match:
        fields["idv"] = float(idv_match.group(1).replace(",", ""))

    ncb_match = NCB_RE.search(text)
    if ncb_match:
        fields["previous_ncb"] = float(ncb_match.group(1))

    seats_match = SEATS_RE.search(text)
    if seats_match:
        fields["seats"] = int(seats_match.group(1))

    gvw_match = GVW_RE.search(text)
    if gvw_match:
        fields["gvw_kg"] = float(gvw_match.group(1).replace(",", ""))

    cc_match = CC_RE.search(text)
    if cc_match:
        fields["engine_cc"] = float(cc_match.group(1))

    engine_match = ENGINE_NO_RE.search(text)
    if engine_match:
        fields["engine_number"] = engine_match.group(1).upper()

    chassis_match = CHASSIS_NO_RE.search(text)
    if chassis_match:
        fields["chassis_number"] = chassis_match.group(1).upper()

    name_match = NAME_RE.search(text)
    if name_match:
        fields["customer_name"] = name_match.group(1).strip()

    vehicle_type = detect_vehicle_type(text)
    if vehicle_type:
        fields["vehicle_type"] = vehicle_type

    fields["_doc_type"] = doc_type
    fields["_raw_text_preview"] = text[:500]
    return fields


def extract_from_upload(file_bytes: bytes, content_type: str) -> dict:
    text = ocr_text_from_file(file_bytes, content_type)
    doc_type = detect_doc_type(text)
    return extract_fields(text, doc_type)


def merge_extracted_fields(field_dicts: list) -> dict:
    """Merge per-document extraction results (e.g. one from an RC book,
    one from a previous policy, or front/back of a smart card RC) into a
    single field set. First non-empty value found wins per key — so if
    one document gives vehicle_type/registration_number and another gives
    idv/previous_ncb, both end up in the merged result rather than one
    document's fields overwriting the other's.

    Also cross-checks registration_number across all documents that have
    one — if they disagree, that's flagged in the returned dict so the
    caller can warn the user before treating the merge as trustworthy,
    rather than silently combining data from what might be two different
    vehicles.

    `_doc_type` and `_raw_text_preview` are collected as lists since
    they're per-document, not a single shared value."""
    merged = {}
    doc_types = []
    previews = []
    reg_numbers_seen = []

    for fields in field_dicts:
        doc_types.append(fields.get("_doc_type", "unknown"))
        previews.append(fields.get("_raw_text_preview", ""))
        reg_no = fields.get("registration_number")
        if reg_no:
            reg_numbers_seen.append(reg_no.strip().upper().replace(" ", ""))

        for key, value in fields.items():
            if key in ("_doc_type", "_raw_text_preview"):
                continue
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value

    merged["_doc_type"] = "+".join(doc_types)
    merged["_raw_text_preview"] = " | ".join(previews)

    distinct_reg_numbers = sorted(set(reg_numbers_seen))
    merged["_registration_numbers_found"] = distinct_reg_numbers
    merged["_registration_mismatch"] = len(distinct_reg_numbers) > 1

    return merged
