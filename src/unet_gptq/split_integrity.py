"""Check only the patient/examination identities actually supplied."""
from __future__ import annotations


def check_calibration_separation(calibration, *, evaluation_files=(), patient_id="", exam_id=""):
    """Reject known overlaps and report the strongest verifiable grouping.

    Equal file hashes reveal overlap, but unequal hashes cannot prove that
    different slice files are from different patients or examinations.
    Missing/partial IDs are never interpreted as disjointness.
    """
    records = calibration.get("data_samples", []) if isinstance(calibration, dict) else []
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("calibration.data_samples must be a list of objects")
    hashes = {str(value) for value in evaluation_files if value}
    for row in records:
        for file in row.get("image_files", []):
            if isinstance(file, dict) and file.get("sha256") in hashes:
                raise ValueError("Calibration and evaluation share a slice file hash")
    for field, current in (("patient_id", patient_id), ("exam_id", exam_id)):
        if current and any(row.get(field) == current for row in records):
            raise ValueError(f"Calibration and evaluation share {field}={current!r}")
    if records and patient_id and all(row.get("patient_id") for row in records):
        return {"status": "verified", "grouping": "patient_id", "n_calibration_pairs": len(records),
                "scope": "supplied patient IDs; their correctness is not independently verified"}
    if records and exam_id and all(row.get("exam_id") for row in records):
        return {"status": "verified", "grouping": "exam_id", "n_calibration_pairs": len(records),
                "scope": "supplied examination IDs; their correctness is not independently verified",
                "limitation": "patient-level disjointness was not established"}
    return {"status": "unverified", "grouping": None, "n_calibration_pairs": len(records),
            "reason": "missing evaluation ID or one or more calibration IDs"}
