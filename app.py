import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore

from ocr_extract import extract_from_upload, merge_extracted_fields
from calc_engine import QuoteInput, calculate_quote
import rate_chart

app = Flask(__name__)
CORS(app)  # allow calls from your GitHub Pages client domain

db = firestore.Client()


def build_quote_input(fields: dict) -> QuoteInput:
    vehicle_type = fields.get("vehicle_type", "Private Cars")
    zone = fields.get("zone", "")
    rating_category = fields.get("rating_category", "")
    vehicle_age_years = int(fields.get("vehicle_age_years") or 0)

    # od_rate_pct / tp_rate / ll_rate_per_psgr always come from the rate
    # chart, not from the request — the chart is the single source of truth.
    # If zone/rating_category aren't set yet (e.g. right after OCR, before
    # the user has picked them), this resolves to a "not matched" rate of 0
    # until the client supplies valid values.
    rate = rate_chart.lookup_rate(vehicle_type, zone, rating_category, vehicle_age_years)

    return QuoteInput(
        vehicle_type=vehicle_type,
        zone=zone,
        rating_category=rating_category,
        vehicle_age_years=vehicle_age_years,
        idv=float(fields.get("idv") or 0),
        previous_ncb=float(fields.get("previous_ncb", -1)),
        seats=fields.get("seats"),
        gvw_kg=fields.get("gvw_kg"),
        od_rate_pct=rate["od_rate_pct"],
        tp_rate=rate["tp_rate"],
        ll_rate_per_psgr=rate["ll_rate_per_psgr"],
        od_discount_pct=float(fields.get("od_discount_pct") or 0),
        addons=fields.get("addons", {}),
    ), rate


CUSTOMER_FIELD_KEYS = [
    "customer_name",
    "customer_address",
    "registration_number",
    "engine_number",
    "chassis_number",
]


def build_customer_fields(source: dict) -> dict:
    """Pull the report-facing identity fields out of whatever dict is
    available (OCR output on first save, or client-submitted edits on
    recalculate) into a flat structure used for admin reporting."""
    return {key: source.get(key, "") for key in CUSTOMER_FIELD_KEYS}


@app.route("/rate-options", methods=["GET"])
def rate_options():
    """Populate the client's zone/category dropdowns with values that
    actually exist in the rate chart, for a given vehicle_type. If a
    `value` query param is given (engine CC, seats, or GVW depending on
    vehicle type), also returns a suggested_category match."""
    vehicle_type = request.args.get("vehicle_type", "")
    zone = request.args.get("zone")
    categories = rate_chart.get_categories(vehicle_type, zone)

    suggested_category = None
    value = request.args.get("value")
    if value:
        suggested_category = rate_chart.match_category(categories, float(value))

    return jsonify({
        "zones": rate_chart.get_zones(vehicle_type),
        "categories": categories,
        "suggested_category": suggested_category,
    })


@app.route("/extract-and-quote", methods=["POST"])
def extract_and_quote():
    """Accepts one or more uploaded files (RC and/or policy) + the logged-in
    user's uid. Runs OCR on each, merges the results, drafts a quote, saves
    it, and returns extracted fields + premium so the client can render an
    editable review screen."""
    uploads = request.files.getlist("file")
    if not uploads:
        return jsonify({"error": "no file uploaded"}), 400

    uid = request.form.get("uid")
    if not uid:
        return jsonify({"error": "missing uid"}), 401

    per_doc_fields = []
    file_errors = []
    for upload in uploads:
        try:
            file_bytes = upload.read()
            content_type = upload.content_type or None
            per_doc_fields.append(extract_from_upload(file_bytes, content_type))
        except Exception as e:
            # Don't let one bad/corrupted file take down the whole request —
            # continue with whatever documents did process, and tell the
            # client which file(s) failed so the user knows to retry them.
            file_errors.append({"filename": upload.filename, "error": str(e)})

    if not per_doc_fields:
        return jsonify({"error": "Could not read any of the uploaded documents", "file_errors": file_errors}), 422

    extracted = merge_extracted_fields(per_doc_fields)

    # OCR doesn't give us zone or the chart's exact rating_category label —
    # those need the user's confirmation (see /rate-options for valid
    # values). Until they're set, rate lookup returns zeros and the quote
    # is a placeholder the client should prompt the user to complete.
    quote_input, rate = build_quote_input(extracted)
    result = calculate_quote(quote_input)
    customer_fields = build_customer_fields(extracted)

    quote_id = str(uuid.uuid4())
    doc = {
        "quote_id": quote_id,
        "uid": uid,
        "product": "motor",
        "vehicle_type": quote_input.vehicle_type,
        "customer_fields": customer_fields,
        "od_discount_pct": quote_input.od_discount_pct,
        "premium": result["final_premium"],
        "extracted_fields": extracted,
        "inputs": quote_input.__dict__,
        "result": result,
        "rate_matched": rate["matched"],
        "status": "draft",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }
    try:
        db.collection("quotes").document(quote_id).set(doc)
    except Exception as e:
        # This is the most likely explanation for "quotes generate fine but
        # nothing shows up in admin" — usually a missing IAM role on the
        # Cloud Run service account (needs "Cloud Datastore User" /
        # "Cloud Firestore User" at minimum) rather than a code bug.
        return jsonify({
            "error": f"Quote calculated but could not be saved: {e}",
            "result": result,
        }), 500

    return jsonify({
        "quote_id": quote_id,
        "extracted_fields": extracted,
        "customer_fields": customer_fields,
        "inputs": quote_input.__dict__,
        "result": result,
        "rate_matched": rate["matched"],
        "rate_candidates": rate["candidates"],
        "file_errors": file_errors,
        "registration_mismatch": extracted.get("_registration_mismatch", False),
        "registration_numbers_found": extracted.get("_registration_numbers_found", []),
    })


@app.route("/recalculate", methods=["POST"])
def recalculate():
    """Takes a quote_id plus any edited fields (od_discount_pct, idv, etc.)
    and re-runs the calc engine only — no OCR. This is what the client's
    'Recalculate' button calls."""
    body = request.get_json(force=True)
    quote_id = body.get("quote_id")
    if not quote_id:
        return jsonify({"error": "missing quote_id"}), 400

    ref = db.collection("quotes").document(quote_id)
    snapshot = ref.get()
    if not snapshot.exists:
        return jsonify({"error": "quote not found"}), 404

    stored = snapshot.to_dict()
    inputs = stored["inputs"]
    inputs.update(body.get("fields", {}))  # apply user edits, e.g. od_discount_pct, idv, zone, rating_category

    customer_fields = stored.get("customer_fields", {})
    customer_fields.update(body.get("customer_fields", {}))

    # Re-run the rate lookup every time — if the user corrected vehicle_type,
    # zone, rating_category, or age, the OD/TP/LL rates need to follow.
    quote_input, rate = build_quote_input(inputs)
    result = calculate_quote(quote_input)

    ref.update({
        "inputs": quote_input.__dict__,
        "customer_fields": customer_fields,
        "od_discount_pct": quote_input.od_discount_pct,
        "premium": result["final_premium"],
        "result": result,
        "rate_matched": rate["matched"],
        "status": "recalculated",
        "updated_at": firestore.SERVER_TIMESTAMP,
    })

    return jsonify({
        "quote_id": quote_id,
        "inputs": quote_input.__dict__,
        "customer_fields": customer_fields,
        "result": result,
        "rate_matched": rate["matched"],
        "rate_candidates": rate["candidates"],
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
