# Telemetry Schema

Every processed chat turn emits one trace event.

Required fields:
- `timestamp_s`
- `session_id`
- `event_type`
- `query`
- `retrieval_required`
- `retrieval_trigger`
- `retrieval_events`
- `citations`
- `answer_version`
- `latency_ms`
- `ttft_ms`
- `prompt_tokens`
- `output_tokens`
- `total_tokens`
- `estimated_cost_usd`
- `uncertainty`
- `metadata`

Retrieval events additionally capture the trigger, query, subqueries, timestamp,
and result count. Token/cost values remain `null` when the provider does not
return usage metadata or no cost rate is configured.
