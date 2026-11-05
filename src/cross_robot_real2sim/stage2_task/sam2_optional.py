"""Optional SAM2 is kept off the critical path and never silently claims masks."""

def segmentation_status(config):
    if not config.get("optional_sam2",False):
        return {"status":"skipped_missing_input","reason":"optional_sam2_disabled; no masks were generated"}
    return {"status":"skipped_missing_input","reason":"SAM2 prompting/weights are not implemented in this MVP; Qwen path remains available"}
