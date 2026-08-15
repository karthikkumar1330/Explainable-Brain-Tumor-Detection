# H7.2 Performance Telemetry Baseline Audit

This audit document establishes the performance telemetry baseline for the AuraScan brain tumor detection and reporting platform, identifying existing diagnostic timings, query performance bottlenecks, request lifecycle tracking, storage architectures, and testing coverage. This is an **audit-only** phase; no production code, test suites, database schemas, or configs have been modified.

---

## 1. Executive Summary

A comprehensive repository-wide audit was conducted to review system latency monitoring, deep learning model execution profiling, REST API request timing, database execution performance, batch prediction tracking, and test coverage.

### Key Findings
1. **N+1 Database Query Loop**: Both the Flask (`/api/history`) and FastAPI (`/database/history`) endpoints execute a synchronous database query (`SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = ? AND patient_id = ?`) inside a loop for every record returned to verify doctor-patient access permissions.
2. **In-Memory Sorting & Filtering**: The prediction history repository fetches all candidates into RAM, performs in-memory PII decryption, executes python-level filtering (such as global query `q` and name matching), and implements pagination/sorting in Python.
3. **No HTTP Request Middleware Timing**: The system lacks middleware to record API request latency, start/end timestamps, HTTP methods, routes, or HTTP status codes.
4. **No Batch Telemetry Persistence**: Batch prediction runs are timed, but statistics (e.g., total batch duration, processing speed, failure rates) are returned only in the API response and are never saved to the database.
5. **Partial Model Identity Mappings**: Loaded model checkpoint file basenames are stored in `ai_audit_logs`, but they are hardcoded strings rather than cryptographic hashes of loaded weights, and are missing from the primary `predictions` table.
6. **No High-Percentile Aggregation**: The system tracks average latency but cannot compute minimum, maximum, median (P50), P95, or P99 latencies without fetching all logs and processing them in Python.

---

## 2. Existing Telemetry Architecture

Performance latency tracking is built into the core diagnostic pipeline:
* **Timeline Tracing**: During report generation, individual pipeline stage elapsed times are tracked in a dictionary (`timeline`) inside `_run_single_report_pipeline()` in [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080).
* **Persistence**:
  * Timelines are saved to the `timeline_traces` table as JSON strings via `db_repo.save_timeline_trace(pred_id, timeline)` in [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L1275-L1288).
  * Total duration is saved as `runtime_sec` in `ai_audit_logs` via `AuditLogger.log_execution()` in [`monitoring/infrastructure/audit_logger.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/monitoring/infrastructure/audit_logger.py#L15-L54).
* **Aggregates**: Historical performance averages are compiled via `get_health_telemetry()` in [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L1197-L1270), calculating average classification execution latency using `SELECT AVG(runtime_sec) FROM ai_audit_logs;`.

We recommend extending the existing SQLite-based `timeline_traces` and `ai_audit_logs` structures rather than introducing a parallel monitoring architecture.

---

## 3. Classification Timing

The classification stage runs inside `_run_single_report_pipeline()` of [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080):
* **Timing Capture**: `t_cls = time.time()` is set on line 522. Preprocessing, model execution (with GPU-to-CPU retry fallback), and Platt calibration run sequentially. Latency is calculated on line 584: `cls_latency = time.time() - t_cls`.
* **Cumulative Timings**: Cumulative durations from request start are captured via `timeline["Classification"]` and `timeline["Calibration"]` on lines 585–586.
* **Gaps**: Isolated timings for classification preprocessing, forward inference pass, and post-inference calibration are **MISSING** (they are bundled inside `cls_latency`). Input validation is measured separately in the calling endpoint (`generate_clinical_report_pipeline`) as `timeline["Validation"]` on line 472.

---

## 4. Segmentation Timing

The segmentation pipeline executes inside `_run_single_report_pipeline()` of [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080):
* **Timing Capture**: `t_seg = time.time()` is set on line 632. Preprocessing, UNeXt inference under autocast, thresholding, and resizing are run. Latency is calculated on line 694: `seg_latency = time.time() - t_seg`.
* **Cumulative Timings**: Cumulative duration is stored in `timeline["Segmentation"]` on line 695.
* **Gaps**: Segmentation preprocessing, forward pass, mask post-processing, and contour/tumor statistics extraction are bundled. Individual preprocessing, inference, and post-processing latencies are **MISSING**. Tumor statistics extraction is timed cumulatively in `timeline["Statistics"]` on line 799, but its standalone duration is not isolated.

---

## 5. Grad-CAM Timing

The explainability pipeline runs inside `_run_single_report_pipeline()` of [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080):
* **Timing Capture**: `t_cam = time.time()` is initialized on line 589. Heatmap generation, target feature registration, text description generation, and overlap calculation run sequentially. Latency is measured on line 710: `cam_latency = time.time() - t_cam`.
* **Cumulative Timings**: Cumulative duration is stored in `timeline["GradCAM"]` on line 711.
* **Gaps**: Image file writing (`cv2.imwrite` for heatmap, overlays, comparison images) is executed *after* `cam_latency` is recorded (lines 713–751). Thus, visual storage and colormap mapping operations are excluded from explainability stage timings.

---

## 6. Total Inference Timing

The system measures total pipeline duration in `_run_single_report_pipeline()` of [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080):
* **Timing Capture**: Calculated on line 822: `total_exec_time = time.time() - t_start`.
* **Calculation Basis**: `t_start` is passed from the routing handler and starts prior to running MRI validation check, mapping all steps sequentially:
  $$\text{Total Runtime} = \text{Validation} + \text{Classification} + \text{Segmentation} + \text{Grad-CAM} + \text{Stats} + \text{PDF Gen} + \text{Email Delivery} + \text{DB Sync} + \text{Notifications} + \text{Auditing}$$
* **Persistence**: Saved to `ai_audit_logs.runtime_sec` via `AuditLogger.log_execution()` on line 1040.

---

## 7. Batch Performance

Batch predictions are handled in `generate_clinical_report_batch()` in [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L1091-L1340):
* **Measured Metrics**:
  * Total batch execution duration: `stats["total_duration_sec"] = time.time() - t_batch_start` on line 1336.
  * Individual file processing latency: `processing_time = time.time() - t_file_start` on lines 1240, 1298, and 1326.
  * Batch sizing: `stats["total_files"]` on line 1133.
  * Execution status counts: valid, invalid, duplicate, successful, and failed counts.
* **Storage Location**: Returned directly in the HTTP JSON response payload. **NOT** persisted in any database table.
* **Gaps**: Stage-specific latencies (classification, segmentation, explainability) are not accumulated or returned at the batch level. Queue/wait time is not tracked as batch predictions are processed in a single-threaded synchronous loop.

---

## 8. API Request Timing

* **FastAPI Middleware**: The Request Telemetry Middleware `add_request_telemetry` inside [`run_api.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/run_api.py#L54-L77) extracts client IP and User-Agent but does **NOT** measure request start, request end, or duration.
* **Flask Middleware**: The `set_audit_context` hook in [`dashboard/infrastructure/web_server.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/dashboard/infrastructure/web_server.py#L120-L135) extracts IP and User-Agent but does **NOT** time requests.
* **Status**:
  * **PARTIAL**: POST `/api/report` (FastAPI) and POST `/api/report/batch` (FastAPI) record internal prediction and file processing times.
  * **MISSING**: HTTP request-level duration, start/end logs, HTTP method, route, and status code logging are completely missing for all endpoints (authentication, report creation, retrieval, scan retrieval, annotation, follow-up, notifications, dashboard APIs, and web views).

---

## 9. Database Performance Telemetry

* **Query Execution Timing**: No query-level execution timings are recorded or logged.
* **Expensive Database Operations**:
  1. **N+1 Access Checks**: Inside both the Flask (`/api/history`) and FastAPI (`/database/history`) routes, the code iterates over all prediction records and executes `auth_svc.can_access_patient()` inside the loop, establishing connections and running queries for every item.
  2. **In-Memory Filtering/Sorting/Decryption**: The `search_history()` method in [`prediction_history/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/prediction_history/infrastructure/repository.py#L44-L268) queries all candidate rows, decrypts patient names in Python using `PIIEncryptionService`, filters results based on RBAC and query parameters (`criteria.q`, `criteria.patient_name`), and performs sorting and pagination in RAM.

---

## 10. Report/PDF Timing

* **Measured Timings**:
  * The PDF generation stage is timed cumulatively in `_run_single_report_pipeline()` of [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L502-L1080) as `timeline["PDF"] = time.time() - t_endpoint_start` on line 917.
  * Standalone PDF compilation time can be calculated as `timeline["PDF"] - timeline["Clinical Report"]`.
* **Gaps**: The system does not isolate Markdown file writing, JSON structure serialization, PDF generation, local storage IO writes, email attachment preparation, or mock SMTP dispatch. SMTP delivery is executed after `timeline["PDF"]` is written and is bundled inside the cumulative `Database` timeline step.

---

## 11. Throughput

* **Current Measurements**: No dedicated metrics or database tables track transaction throughput.
* **Queryability**: Throughput (predictions per hour or day) can be derived by querying `ai_audit_logs` using:
  ```sql
  SELECT COUNT(*), strftime('%Y-%m-%d %H', timestamp) FROM ai_audit_logs GROUP BY 2;
  ```
  Batch throughput is calculated dynamically during batch execution but is not stored.

---

## 12. Error Timing

* **Current Measurements**:
  * If a pipeline stage fails (e.g. classification, segmentation, or Grad-CAM), the failure is caught by `PipelineExecutionRecovery` and default fallbacks are applied.
  * The overall execution time (`total_exec_time` or batch item `processing_time`) is still measured and recorded.
  * However, the duration elapsed *until* failure is not isolated.
  * The exception categories are stored in `ai_audit_logs.errors_json` and logged in server logs.

---

## 13. Resource Correlation

* **Recorded Parameters**: `ai_audit_logs` records `gpu_active` (1/0) and `cpu_threads` (multiprocessing core count) at the time of prediction writing.
* **Pipeline health checks**: Static system specs (CPU cores, threads) and current hardware usage (RAM usage %, disk usage %, VRAM allocated/total) are fetched dynamically by `PipelineHealthMonitor.get_system_metrics()` in [`monitoring/infrastructure/health_monitor.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/monitoring/infrastructure/health_monitor.py#L38-L108) during `/health-telemetry` requests.
* **Gaps**: Dynamic CPU, RAM, and GPU/VRAM utilisation *during* the execution of a specific prediction run is **NOT** captured or correlated with individual prediction timings.

---

## 14. Model Version Correlation

* **Model Checkpoints**: Basenames of checkpoint files (`efficientnet_b0_brain_tumor.pth` and `model.pth`) are written to `ai_audit_logs` as `model_version_cls` and `model_version_seg` in [`api/infrastructure/routes.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/api/infrastructure/routes.py#L1043-L1044).
* **Gaps**:
  1. Cryptographic hashes of checkpoint weights are not validated.
  2. Model version information is absent from the core `predictions` table, risking loss of model identity correlation if audit logs are pruned.
  3. No semantic versions (e.g., v1.0.0) are tracked.

---

## 15. Storage Architecture

Performance metrics are stored in SQLite tables:
1. **`timeline_traces`**:
   - **Schema**: Defined in [`persistence/infrastructure/repository.py#L141-L148`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L141-L148)
     - `prediction_id INTEGER PRIMARY KEY` (FK to `predictions.id` with `ON DELETE CASCADE`)
     - `trace_json TEXT NOT NULL` (JSON-string of pipeline step cumulative latencies)
     - `created_at TEXT NOT NULL` (ISO timestamp)
   - **Indexes**: None.
2. **`ai_audit_logs`**:
   - **Schema**: Defined in [`persistence/infrastructure/repository.py#L121-L139`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L121-L139)
     - `runtime_sec REAL NOT NULL` (total inference duration)
     - `gpu_active INTEGER NOT NULL`
     - `cpu_threads INTEGER NOT NULL`
     - `warnings_json TEXT NOT NULL`
     - `errors_json TEXT NOT NULL`
   - **Indexes**:
     - `idx_ai_audit_logs_timestamp` (on `timestamp`) on line 365.
     - `idx_ai_audit_logs_patient` (on `patient_id`) on line 366.

---

## 16. Aggregation Capabilities

* **CURRENTLY AVAILABLE**: Average pipeline execution latency, total prediction counts, diagnosis distributions, and duplicate upload frequencies are compiled via `get_health_telemetry()` in [`persistence/infrastructure/repository.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/persistence/infrastructure/repository.py#L1197-L1273).
* **REQUIRES IMPLEMENTATION**: Minimum, maximum, median (P50), P90, P95, P99, and standard deviation calculations are **MISSING** (they are not supported by SQLite's default query compiler without new custom function registrations or loading data into memory in Python).

---

## 17. Dashboard Usage

* **Displayed Metrics**: The average classification execution latency (compiled as `avg_runtime` in `get_health_telemetry()`) is returned to the frontend.
* **Role Boundary**: Telemetry data is restricted to the `/api/health-telemetry` route, which enforces `@roles_accepted(Role.ADMIN)`. Doctors and patients cannot access this infrastructure performance telemetry.
* **Real-time vs. Historical**: The metrics are historical averages computed from the SQLite database.

---

## 18. Existing Test Coverage

Existing test suites validate telemetry, audit logs, and performance indexes:
1. **[`tests/test_h54_telemetry.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/tests/test_h54_telemetry.py)**:
   - `test_01_flask_propagates_request_telemetry`: Verifies that Flask propagates client IP and User-Agent to the audit context.
   - `test_02_fastapi_propagates_request_telemetry`: Verifies that FastAPI propagates client IP and User-Agent.
   - `test_03_background_job_falls_back_safely`: Verifies background threads run safely without HTTP request context.
   - `test_04_proxy_ip_handling_based_on_trust`: Verifies `X-Forwarded-For` header extraction depending on `TRUST_PROXY` config.
2. **[`tests/test_operations.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/tests/test_operations.py)**:
   - `test_timeline_persistence`: Verifies writing and loading step latencies from `timeline_traces` table.
   - `test_model_version_manager`: Verifies checkpoint version tracking mapping.
   - `test_pipeline_recovery_graceful_fallback`: Verifies recovery fallback works under execution failure.
3. **[`tests/test_rbac_integration.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/tests/test_rbac_integration.py)**:
   - `test_health_telemetry_endpoint_rbac`: Verifies only `ADMIN` users can query detailed system telemetry.
4. **[`tests/test_h64_optimization.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/tests/test_h64_optimization.py)**:
   - `test_06_explain_query_plan_index_usage`: Runs `EXPLAIN QUERY PLAN` in SQLite to verify database index utilization and prevent full table scans.
5. **[`tests/test_h71_monitoring_security.py`](file:///d:/BrainTumorProject/UNeXt-pytorch/tests/test_h71_monitoring_security.py)**:
   - `test_01_health_is_public_and_minimal`: Verifies `/health` is unauthenticated and returns minimal information.
   - `test_02_health_does_not_leak_telemetry`: Verifies `/health` does not leak hardware specs, IP addresses, or pathnames.
   - `test_04_admin_can_access_detailed_telemetry`: Verifies `ADMIN` can fetch detailed telemetry on `/health-telemetry`.
   - `test_05_doctor_is_denied_detailed_telemetry`: Verifies `DOCTOR` is denied access to detailed system telemetry (returns 403 Forbidden).

---

## 19. Security Review

* **PII & Infrastructure Separation**: Existing performance telemetry does **NOT** expose patient names, MRI raw data paths, report markdown contents, SMTP passwords, encryption keys, or SQL credentials. Patient names are stored in encrypted format inside the `patients` table.
* **Endpoint Hardening**: Public endpoints `/health` and `/ping` are stripped of performance telemetry to prevent fingerprinting. System hardware properties (CPU/VRAM/Uptime) are accessible ONLY to the `ADMIN` role via `/health-telemetry`.

---

## 20. H7.2 Metric Gap Matrix

| Metric | Existing? | Stored? | Queryable? | Aggregatable? | Displayed? | Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **preprocessing latency** | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Preprocessing bundled inside general stage timing. |
| **classification latency** | 🟢 YES | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Stored in JSON blob, not indexed or SQL aggregatable. |
| **segmentation latency** | 🟢 YES | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Stored in JSON blob, not indexed or SQL aggregatable. |
| **postprocessing latency** | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Modular post-processing not timed. |
| **Grad-CAM latency** | 🟢 YES | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Stored in JSON blob, not indexed or SQL aggregatable. |
| **total inference latency** | 🟢 YES | 🟢 YES | 🟢 YES | 🟢 YES | 🟢 YES | Persisted in `ai_audit_logs.runtime_sec` and aggregated. |
| **batch duration** | 🟢 YES | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Returned in API response, not saved in database. |
| **per-image duration** | 🟢 YES | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Not saved for failed batch items. |
| **API request latency** | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | No HTTP middleware timing exists. |
| **report generation latency**| 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Grouped inside PDF timeline step. |
| **PDF generation latency** | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Bundled inside report generation step. |
| **database query latency** | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Database query execution times are not measured. |
| **throughput** | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | Calculated dynamically from logs but no dedicated view. |
| **error duration** | 🟡 PARTIAL | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO | Standalone time elapsed until fallback failure is missing. |
| **CPU correlation** | 🟡 PARTIAL | 🟡 PARTIAL | 🟡 PARTIAL | 🔴 NO | 🟡 PARTIAL | Static specs stored, active execution load missing. |
| **RAM correlation** | 🟡 PARTIAL | 🔴 NO | 🔴 NO | 🔴 NO | 🟡 PARTIAL | RAM load not correlated with runtime. |
| **GPU correlation** | 🟢 YES | 🟢 YES | 🟢 YES | 🟢 YES | 🔴 NO | `gpu_active` stored in audit logs but not dashboarded. |
| **model-version correlation**| 🟢 YES | 🟢 YES | 🟢 YES | 🟢 YES | 🔴 NO | Filenames stored in audit logs, missing in predictions. |

---

## 21. COMPLETE capabilities

* **Diagnostic Stage Timings**: Individual classification (`cls_latency`), explainability (`cam_latency`), and segmentation (`seg_latency`) durations are fully measured inside the prediction pipeline.
* **Total Runtime Persistence**: Total inference execution duration (`runtime_sec`) is recorded and aggregated to average metrics.
* **Basic Hardware Specs Check**: Active GPU usage (`gpu_active`) and CPU threads count are successfully correlated with diagnostic logs.

---

## 22. PARTIAL capabilities

* **Pipeline Stage Timing Queries**: Timings are saved to `timeline_traces`, but inside a JSON string blob (`trace_json`). This prevents direct indexing and SQL queries like `AVG()` or `MAX()`.
* **Model Checkpoint Tracking**: The checkpoint file basenames are saved in audit logs, but they are hardcoded strings, not cryptographic hashes of loaded parameters, and are not stored in the `predictions` table.
* **Batch Sizing and Speed**: Batch metrics are computed during requests and returned to clients but are not stored in the database.

---

## 23. MISSING capabilities

* **HTTP Middleware Timing**: The system has no HTTP request-level logging for endpoint latency, routes, or response status codes.
* **Database Query Performance Telemetry**: Profiling of SQL query executions is missing.
* **Percentile Aggregations**: The database is unable to run P50, P90, P95, or P99 latency aggregations without parsing JSON blobs or fetching rows in Python.
* **Preprocessing & Postprocessing Timings**: Input image resize/normalize operations are not timed.

---

## 24. MUST implementation items

1. **Database Query Optimization**: Resolve the N+1 database query loop in history routes (`/api/history` and `/database/history`) by pre-fetching patient assignments or executing a join query.
2. **Request Timing Middleware**: Implement a non-intrusive HTTP middleware in Flask and FastAPI to record request duration, HTTP method, route, and status code, printing to system logs.
3. **Batch Prediction Persistence**: Save batch prediction metadata (batch ID, total images, successful count, failed count, total duration, start time) to a new SQLite database table.

---

## 25. HIGH priority items

1. **Database Stage Timings Indexing**: Extract classification, segmentation, and Grad-CAM timings from the JSON blob and persist them as separate SQL columns inside `timeline_traces` (or `ai_audit_logs`) to support indexing and aggregations.
2. **Percentile Queries Support**: Add SQL-level support or Python repository functions to calculate median, P90, P95, and P99 latency statistics.

---

## 26. MEDIUM/LOW items

1. **Active Hardware Load Profiling**: Track memory/CPU/VRAM usage dynamically during model forward passes.
2. **Cryptographic Model Hashing**: Validate checkpoints using SHA-256 weight parameters verification.

---

## 27. Recommended H7.2 implementation order

1. **Phase 1: HTTP Request Telemetry Middleware**: Set up a non-blocking latency middleware to log endpoint metrics.
2. **Phase 2: N+1 History Optimization**: Implement Join-based authorization checks in historical searches to replace the looping checks.
3. **Phase 3: Database Timings Column Re-Structuring**: Move pipeline timings from JSON blobs to structured SQL table columns to support SQL aggregations.
4. **Phase 4: Batch Metatables Schema**: Build batch-level database tables and persist summary stats on execution.

---

## 28. H7.2 test strategy

* **Unit Tests**:
  * Assert request timing middleware prints logs with expected HTTP status and route mappings.
  * Verify that optimizing history query results in a single database execution block instead of N queries (via `Mock` calls count).
* **Integration Tests**:
  * Verify that batch prediction logs summary statistics inside the batch database table.
  * Verify role-based boundaries on telemetry metrics are preserved.

---

## 29. H7.2 regression strategy

* **Intact Security Logic**: All H7.1 security boundaries (denying DOCTOR/PATIENT access to infrastructure endpoints, preventing stack trace leaks, enforcing csrf validation) must remain unmodified.
* **Inference Preservation**: Model weight calculations and fallback structures must not be altered.
* **Verification**: Run `pytest tests/` after any changes to verify 100% of the 733 existing tests continue to pass.

---

## 30. H7.2 completion gate

1. `H7_2_PERFORMANCE_TELEMETRY_BASELINE_AUDIT.md` is present in the workspace root.
2. No functional modifications have been made to production code files.
3. All existing unit and integration tests execute successfully.
4. Working tree remains clean.
