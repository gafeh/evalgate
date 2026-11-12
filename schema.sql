BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 72c049199793

CREATE TABLE datasets (
    name VARCHAR(128) NOT NULL, 
    version VARCHAR(64) NOT NULL, 
    domain VARCHAR(64) NOT NULL, 
    content_hash VARCHAR(64) NOT NULL, 
    case_count INTEGER NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_datasets PRIMARY KEY (id), 
    CONSTRAINT uq_datasets_name_version UNIQUE (name, version)
);

CREATE INDEX ix_datasets_content_hash ON datasets (content_hash);

CREATE TABLE cases (
    dataset_id UUID NOT NULL, 
    external_id VARCHAR(128) NOT NULL, 
    question_id VARCHAR(128) NOT NULL, 
    question_prompt TEXT NOT NULL, 
    student_response TEXT NOT NULL, 
    rubric JSONB NOT NULL, 
    expected JSONB NOT NULL, 
    tags JSONB NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_cases PRIMARY KEY (id), 
    CONSTRAINT fk_cases_dataset_id_datasets FOREIGN KEY(dataset_id) REFERENCES datasets (id) ON DELETE CASCADE, 
    CONSTRAINT uq_cases_dataset_id_external_id UNIQUE (dataset_id, external_id)
);

CREATE INDEX ix_cases_dataset_id ON cases (dataset_id);

CREATE INDEX ix_cases_question_id ON cases (question_id);

CREATE TABLE runs (
    dataset_id UUID NOT NULL, 
    dataset_content_hash VARCHAR(64) NOT NULL, 
    model VARCHAR(128) NOT NULL, 
    params JSONB NOT NULL, 
    label VARCHAR(128), 
    status VARCHAR(16) NOT NULL, 
    started_at TIMESTAMP WITH TIME ZONE, 
    finished_at TIMESTAMP WITH TIME ZONE, 
    error TEXT, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_runs PRIMARY KEY (id), 
    CONSTRAINT fk_runs_dataset_id_datasets FOREIGN KEY(dataset_id) REFERENCES datasets (id) ON DELETE RESTRICT
);

CREATE INDEX ix_runs_dataset_id ON runs (dataset_id);

CREATE INDEX ix_runs_status ON runs (status);

CREATE TABLE results (
    run_id UUID NOT NULL, 
    case_id UUID NOT NULL, 
    raw_output TEXT NOT NULL, 
    parsed_ok BOOLEAN NOT NULL, 
    parse_error TEXT, 
    grading JSONB, 
    scores JSONB NOT NULL, 
    overall FLOAT, 
    over_credit_points FLOAT, 
    under_credit_points FLOAT, 
    hallucinated_item_ids JSONB NOT NULL, 
    latency_ms INTEGER, 
    content_hash VARCHAR(64) NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_results PRIMARY KEY (id), 
    CONSTRAINT fk_results_case_id_cases FOREIGN KEY(case_id) REFERENCES cases (id) ON DELETE CASCADE, 
    CONSTRAINT fk_results_run_id_runs FOREIGN KEY(run_id) REFERENCES runs (id) ON DELETE CASCADE, 
    CONSTRAINT uq_results_run_id_case_id UNIQUE (run_id, case_id)
);

CREATE INDEX ix_results_run_id ON results (run_id);

CREATE TABLE work_items (
    run_id UUID NOT NULL, 
    case_id UUID NOT NULL, 
    status VARCHAR(16) NOT NULL, 
    attempts INTEGER NOT NULL, 
    max_attempts INTEGER NOT NULL, 
    claimed_by VARCHAR(128), 
    claimed_at TIMESTAMP WITH TIME ZONE, 
    heartbeat_at TIMESTAMP WITH TIME ZONE, 
    last_error TEXT, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_work_items PRIMARY KEY (id), 
    CONSTRAINT fk_work_items_case_id_cases FOREIGN KEY(case_id) REFERENCES cases (id) ON DELETE CASCADE, 
    CONSTRAINT fk_work_items_run_id_runs FOREIGN KEY(run_id) REFERENCES runs (id) ON DELETE CASCADE, 
    CONSTRAINT uq_work_items_run_id_case_id UNIQUE (run_id, case_id)
);

CREATE INDEX ix_work_items_claimable ON work_items (run_id, status);

INSERT INTO alembic_version (version_num) VALUES ('72c049199793') RETURNING alembic_version.version_num;

-- Running upgrade 72c049199793 -> c9c66d9b0ea2

ALTER TABLE work_items ADD COLUMN available_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL;

DROP INDEX ix_work_items_claimable;

CREATE INDEX ix_work_items_claimable ON work_items (run_id, status, available_at);

UPDATE alembic_version SET version_num='c9c66d9b0ea2' WHERE alembic_version.version_num = '72c049199793';

-- Running upgrade c9c66d9b0ea2 -> f6f8d80790c8

CREATE TABLE llm_invocations (
    cache_key VARCHAR(64) NOT NULL, 
    purpose VARCHAR(16) NOT NULL, 
    model VARCHAR(128) NOT NULL, 
    request JSONB NOT NULL, 
    response_payload JSONB, 
    raw_text TEXT NOT NULL, 
    parse_error TEXT, 
    input_tokens INTEGER NOT NULL, 
    output_tokens INTEGER NOT NULL, 
    cost_usd FLOAT NOT NULL, 
    latency_ms INTEGER NOT NULL, 
    stop_reason VARCHAR(64), 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_llm_invocations PRIMARY KEY (id), 
    CONSTRAINT uq_llm_invocations_cache_key UNIQUE (cache_key)
);

CREATE TABLE llm_calls (
    run_id UUID NOT NULL, 
    purpose VARCHAR(16) NOT NULL, 
    model VARCHAR(128) NOT NULL, 
    cache_key VARCHAR(64) NOT NULL, 
    cached BOOLEAN NOT NULL, 
    input_tokens INTEGER NOT NULL, 
    output_tokens INTEGER NOT NULL, 
    cost_usd FLOAT NOT NULL, 
    latency_ms INTEGER NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_llm_calls PRIMARY KEY (id), 
    CONSTRAINT fk_llm_calls_run_id_runs FOREIGN KEY(run_id) REFERENCES runs (id) ON DELETE CASCADE
);

CREATE INDEX ix_llm_calls_cache_key ON llm_calls (cache_key);

CREATE INDEX ix_llm_calls_run_purpose ON llm_calls (run_id, purpose);

CREATE TABLE judge_assessments (
    result_id UUID NOT NULL, 
    model VARCHAR(128) NOT NULL, 
    verdict VARCHAR(16) NOT NULL, 
    confidence FLOAT NOT NULL, 
    reasoning TEXT NOT NULL, 
    criteria JSONB NOT NULL, 
    cache_key VARCHAR(64) NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_judge_assessments PRIMARY KEY (id), 
    CONSTRAINT fk_judge_assessments_result_id_results FOREIGN KEY(result_id) REFERENCES results (id) ON DELETE CASCADE, 
    CONSTRAINT uq_judge_assessments_result_id UNIQUE (result_id)
);

UPDATE alembic_version SET version_num='f6f8d80790c8' WHERE alembic_version.version_num = 'c9c66d9b0ea2';

-- Running upgrade f6f8d80790c8 -> 79a54d50f49b

UPDATE alembic_version SET version_num='79a54d50f49b' WHERE alembic_version.version_num = 'f6f8d80790c8';

-- Running upgrade 79a54d50f49b -> fa8fa83582af

CREATE TABLE review_items (
    result_id UUID NOT NULL, 
    status VARCHAR(16) NOT NULL, 
    reason VARCHAR(16) NOT NULL, 
    priority FLOAT NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_review_items PRIMARY KEY (id), 
    CONSTRAINT fk_review_items_result_id_results FOREIGN KEY(result_id) REFERENCES results (id) ON DELETE CASCADE, 
    CONSTRAINT uq_review_items_result_id UNIQUE (result_id)
);

CREATE INDEX ix_review_items_pending ON review_items (status, priority);

CREATE TABLE human_labels (
    review_item_id UUID NOT NULL, 
    labeller VARCHAR(64) NOT NULL, 
    verdict VARCHAR(16) NOT NULL, 
    criteria JSONB NOT NULL, 
    notes TEXT NOT NULL, 
    seconds_spent INTEGER NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_human_labels PRIMARY KEY (id), 
    CONSTRAINT fk_human_labels_review_item_id_review_items FOREIGN KEY(review_item_id) REFERENCES review_items (id) ON DELETE CASCADE, 
    CONSTRAINT uq_human_labels_review_item_id UNIQUE (review_item_id)
);

CREATE INDEX ix_human_labels_labeller ON human_labels (labeller);

UPDATE alembic_version SET version_num='fa8fa83582af' WHERE alembic_version.version_num = '79a54d50f49b';

-- Running upgrade fa8fa83582af -> 8e1c9915624e

CREATE TABLE audit_entries (
    run_id UUID NOT NULL, 
    seq INTEGER NOT NULL, 
    entry_type VARCHAR(32) NOT NULL, 
    payload JSONB NOT NULL, 
    prev_hash VARCHAR(64) NOT NULL, 
    entry_hash VARCHAR(64) NOT NULL, 
    id UUID NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    CONSTRAINT pk_audit_entries PRIMARY KEY (id), 
    CONSTRAINT fk_audit_entries_run_id_runs FOREIGN KEY(run_id) REFERENCES runs (id) ON DELETE CASCADE, 
    CONSTRAINT uq_audit_entries_run_id_seq UNIQUE (run_id, seq)
);

CREATE INDEX ix_audit_entries_entry_hash ON audit_entries (entry_hash);

UPDATE alembic_version SET version_num='8e1c9915624e' WHERE alembic_version.version_num = 'fa8fa83582af';

COMMIT;

