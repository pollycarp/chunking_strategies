-- Migration 005: Agent interaction logs
--
-- Why a separate table and not application logs?
-- Structured DB storage lets us query, aggregate, and trend quality metrics
-- over time. We can answer: "Has average latency increased since Monday?" or
-- "What fraction of answers had hallucinations detected last week?" — things
-- that are impossible to answer from flat log files without parsing them.
--
-- Columns:
--   tools_called        : JSONB array of tool names used in the interaction
--   latency_ms          : wall-clock time from question received to answer sent
--   input_tokens        : tokens in the prompt (question + history + context)
--   output_tokens       : tokens in the generated answer
--   faithfulness_score  : 0.0–1.0 from LLM-as-judge (NULL if not evaluated)
--   hallucination_detected : True if any numeric claim was ungrounded (NULL if not checked)

CREATE TABLE IF NOT EXISTS agent_logs (
    id                      BIGSERIAL    PRIMARY KEY,
    logged_at               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    question                TEXT         NOT NULL,
    answer                  TEXT         NOT NULL,
    tools_called            JSONB        NOT NULL DEFAULT '[]',
    latency_ms              INTEGER      NOT NULL,
    input_tokens            INTEGER      NOT NULL DEFAULT 0,
    output_tokens           INTEGER      NOT NULL DEFAULT 0,
    faithfulness_score      FLOAT,
    hallucination_detected  BOOLEAN
);

-- Most queries will be "show me the last N interactions" or
-- "show me stats for the last 24 hours" — index on time descending.
CREATE INDEX IF NOT EXISTS agent_logs_logged_at_idx
    ON agent_logs (logged_at DESC);
