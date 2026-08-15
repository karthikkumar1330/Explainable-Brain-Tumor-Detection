# H7.2-D Model Identity / Provenance Baseline Audit

## 1. Executive Summary
This baseline audit evaluates the tracking of model identity and provenance across the brain tumor classification, segmentation, and explainability pipelines. Currently, the system executes predictions without persistently recording which exact model, architecture version, or checkpoint hash was utilized. This gap makes it impossible to answer "Which exact model produced this prediction?" with absolute cryptographic or queryable certainty.

## 2. Existing Model-Loading Architecture
Models are preloaded during API startup via the `initialize_api_models()` function inside [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L63).
- **Classification Model**: Instantiated as `EfficientNetB0Model` from [`classification/infrastructure/models.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/classification/infrastructure/models.py#L14) and wrapped in a `PyTorchModelAdapter`. Checkpoint loaded from `CLS_CHECKPOINT = "models/classification/efficientnet_b0_brain_tumor.pth"`.
- **Segmentation Model**: Dynamically loaded using YAML configuration parameter `SEG_CONFIG = "models/brain_tumor_unext/config.yml"` and architecture instantiation from `archs.__dict__` in [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L90). Checkpoint loaded from `SEG_CHECKPOINT = "models/brain_tumor_unext/model.pth"`.
- **Explainability**: Instantiated dynamically on execution using `PyTorchXAIEngine` hook mappings in [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L602) targeting classification layer features.

---

## 3. Classification Model Provenance
- **Model Name**: `efficientnet_b0` (or dynamically from `ModelRegistry` in ensemble mode).
- **Architecture**: `EfficientNet-B0`.
- **Checkpoint Filename**: `efficientnet_b0_brain_tumor.pth`
- **Checkpoint Path**: `models/classification/efficientnet_b0_brain_tumor.pth`
- **Model Version**: 🔴 MISSING. No model version configuration exists.
- **Checkpoint Hash**: 🔴 MISSING. Checked neither on startup nor on prediction.
- **Loading Timestamp**: 🔴 MISSING.
- **Device**: Dynamically allocated (`cuda` or `cpu` fallback).
- Persisted in DB: 🔴 MISSING.
- Included in Audit Logs: 🔴 MISSING.
- Included in Prediction Records: 🔴 MISSING.

---

## 4. Segmentation Model Provenance
- **Model Name**: `brain_tumor_unext`
- **Architecture**: `UNeXt` (instantiated dynamically from YAML configurations).
- **Checkpoint Filename**: `model.pth`
- **Checkpoint Path**: `models/brain_tumor_unext/model.pth`
- **Model Version**: 🔴 MISSING.
- **Checkpoint Hash**: 🔴 MISSING.
- **Loading Timestamp**: 🔴 MISSING.
- **Device**: Dynamically allocated (`cuda` or `cpu` fallback).
- Persisted in DB: 🔴 MISSING.
- Included in Audit Logs: 🔴 MISSING.
- Included in Prediction Records: 🔴 MISSING.

---

## 5. Explainability Model Provenance
- **Model Name**: `gradcam`, `gradcam_plus_plus`, or `eigencam` (based on user intake parameter).
- **Architecture**: PyTorch forward hooks.
- **Checkpoint Filename/Path**: Inherited from Classification Model.
- **Model Version**: 🔴 MISSING.
- **Checkpoint Hash**: 🔴 MISSING.
- **Device**: Inherited from Classification Model.
- Persisted in DB: 🟡 PARTIAL. The text string of the selected `xai_method` is saved in `clinical_reports` (e.g. `"gradcam"`).
- Included in Audit Logs: 🔴 MISSING.
- Included in Prediction Records: 🔴 MISSING.

---

## 6. Existing Audit-Log Provenance
- **Table**: `security_audit_logs` in [`security/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/security/infrastructure/repository.py#L121).
- **Provenance info**: 🔴 MISSING. The table records high-level workflow events (e.g. `LOGIN_SUCCESS`, `MRI_UPLOAD`, `REPORT_VIEWED`) but logs absolutely no model loading, execution metadata, device types, or checkpoint hashes.

---

## 7. Existing Prediction Provenance
- **Table**: `predictions` in [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L70).
- **Provenance info**: 🔴 MISSING. The table stores only labels, confidence scores, probability distributions, severity levels, and timestamps. It does not record the active model's name, version, hash, or execution device.

---

## 8. Existing Batch Telemetry Provenance
- **Table**: `batch_performance_telemetry` in [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L367).
- **Provenance info**: 🔴 MISSING. Records only counts, timestamp, and latencies. No model identifiers are logged.

---

## 9. Existing Model Version/Checkpoint Handling
- **Reloading**: Global model holders are preloaded into memory at startup. There is no automated reinitialization or auto-recovery validation if the model checkpoint file is modified on disk.
- **Failure handling**: Implements graceful fallback to CPU if GPU inference crashes, but the device change is not written to database records.

---

## 10. Security Analysis
- 🔴 **SECURITY GAP**: The `ProcessingSummary` struct generated during integration reports absolute local filesystem paths (`models/classification/...`). Exposing absolute file paths or system-level paths to users via APIs is a disclosure vulnerability.
- **Mitigation**: Standardize on relative normalized identifiers (e.g., relative to model folder root) and filter out absolute paths.

---

## 11. Performance Impact Analysis
- Calculating the SHA-256 checkpoint checksum on every inference request would introduce high latency (~30-50ms overhead per prediction).
- **Optimization**: Checksum calculations must be executed **only once at startup (warmup)** during `initialize_api_models()` and stored in an in-memory cache, ensuring zero latency impact during request executions.

---

## 12. Gap Analysis

| Gap / Gaps | Classification / Location | Severity / Priority | Current Behavior | Recommended Solution |
| :--- | :--- | :--- | :--- | :--- |
| **Model metadata not tracked** | `persistence/infrastructure/repository.py` | 🔴 MISSING / MUST | `predictions` and `clinical_reports` store no model details. | Add `model_provenance` table and reference it via FK from `predictions`. |
| **Verification of checkpoints** | `api/infrastructure/routes.py` | 🟠 IMPROVEMENT / HIGH | Checkpoint integrity is not verified on loading. | Generate SHA-256 hashes of models once at startup. |
| **Logs lack model provenance** | `security/infrastructure/repository.py` | 🔴 MISSING / MEDIUM | Audit logs omit execution info. | Include model execution info in prediction details. |
| **Ensemble Mode mapping missing** | `api/infrastructure/routes.py` | 🟡 PARTIAL / HIGH | Ensemble mode calculates multiple models but writes only one final result. | Log all models active in the ensemble via a junction mapping table or serialized list. |
| **Absolute Path Leak** | `clinical_reporting/domain/entities.py` | 🔴 SECURITY GAP / MUST | Exposes local checkpoint path. | Sanitize path representation (relative or masked). |

---

## 13. Recommended Architecture
We select **Architecture C + B** (Dedicated `model_provenance` table + Prediction record linking):
- A new table `model_provenance` will act as a registry of loaded models and state dict signatures.
- Startup model initialization will calculate SHA-256 hashes, register them in `model_provenance`, and return their registered `id`.
- The `predictions` table will be updated with `classification_model_provenance_id` and `segmentation_model_provenance_id` foreign keys linking to the exact executing model registry record.

---

## 14. Required Schema Changes
Add the `model_provenance` table schema:
```sql
CREATE TABLE IF NOT EXISTS model_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,         -- 'CLASSIFICATION' or 'SEGMENTATION'
    model_name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,    -- Relative path only
    checkpoint_hash TEXT NOT NULL,    -- SHA-256
    device TEXT NOT NULL,             -- 'cpu' or 'cuda'
    loaded_at TEXT NOT NULL
);
```

Add foreign keys to the `predictions` table:
- `classification_model_provenance_id` (INTEGER, FOREIGN KEY to `model_provenance(id)`)
- `segmentation_model_provenance_id` (INTEGER, FOREIGN KEY to `model_provenance(id)`)

---

## 15. Required Code Changes
1. [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py): Add `model_provenance` schema creation, index, and insertion methods. Extend `save_report()` to link predictions to active model ids.
2. [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py): Add startup SHA-256 hash generation for `CLS_CHECKPOINT` and `SEG_CHECKPOINT`. Register models in the `model_provenance` table and capture their IDs. Pass these IDs to prediction persistence calls.
3. [`clinical_reporting/domain/entities.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/clinical_reporting/domain/entities.py): Sanitize output file paths in `ProcessingSummary` to report relative paths instead of absolute system paths.

---

## 16. Required Tests
Create `tests/test_h72d_model_provenance.py` covering:
1. Startup calculation of SHA-256 matches model checkpoints.
2. Successful model registration inside `model_provenance` table.
3. Predictions successfully link to their corresponding executing model records via FK.
4. CPU fallback execution updates the loaded model device column in `model_provenance` or logs the fallback event details.
5. No absolute paths are exposed through the public endpoints.
6. Checkpoint modification/tampering is detected or logged.

---

## 17. Implementation Order
1. Implement SHA-256 checksum helper in routes initialization.
2. Add `model_provenance` table schema and SQL migrations in `repository.py`.
3. Update `save_report` database insertion code to link prediction rows to model provenance.
4. Sanitize `ProcessingSummary` absolute path outputs.
5. Create and execute tests.

---

## 18. H7.2-D Completion Gate
- All tests in `tests/test_h72d_model_provenance.py` pass.
- Full regression suite execution is 100% green.
- Git status remains clean and git check passes with zero warnings.
