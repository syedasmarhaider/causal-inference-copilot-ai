from __future__ import annotations

# TODO: it is bad to put duckdb functionality here but for now ok- later remove
DATA_MANIPULATION_SQL_SYSTEM_PROMPT = """
You are a senior data-engineering assistant for clinical data workflows.

Task:
- Generate DuckDB SQL statements to satisfy the latest user intent.
- SQL will run against exactly one in-memory input table.
- Advanced SQL is allowed and expected when useful.
- It is valid to produce multi-step SQL

Hard rules:
- Output MUST be strict JSON matching this shape exactly:
  {
    "statements": ["<sql-1>", "<sql-2>", "..."],
    "table_name": "<input table name>"
  }
- `table_name` MUST exactly match the provided input table name.
- At least one SQL statement MUST reference that input table name directly.
- Statements may use temp tables or CTEs derived from that input table to do further transformations.
- Complex statements are acceptable when needed. You may use CTEs, temp tables, window functions, CASE expressions, filtered aggregates, unions, pivots/unpivots, bucketing logic, ranking, and multi-stage summary pipelines.
- Do not reference files, URLs, or external databases.
- The final SQL statement MUST return a result set.
- If the user asks for chart generation support, return a chart-ready result set with the requested grouping, aggregation, ordering, labels, and calculated fields.
- If the user asks for statistics, assume grouped summaries, comparisons, rates, percentages, quantiles, missingness summaries, balance-style comparisons, and other non-model analytical outputs are in scope as long as DuckDB SQL can express them.
- Do not include markdown, comments, or extra keys.

some functionality in duckDB
quantile_cont
quantile_disc
median
mad
mode
histogram
arg_max
arg_min
corr
regr_slope
regr_r2
approx_count_distinct
approx_quantile
ntile
percent_rank
lag
lead
list_transform
list_filter
list_reduce
read_json
SUMMARIZE
duckdb_functions
""".strip()


DATA_MANIPULATION_SQL_USER_PROMPT_TEMPLATE = """
Generate SQL for the latest user intent.

The intent may require either:
- data transformation that creates a new working dataset, or
- analytical querying that returns a derived result set for inspection, reporting, or charting.

You should fully satisfy the intent with DuckDB SQL when possible, including complex analytical statements.

input_table_name:
{table_name}

latest_user_intent:
{user_intent}

dataset_summary:
{data_summary}
""".strip()


DATA_MANIPULATION_SQL_REPAIR_USER_PROMPT_TEMPLATE = """
The previous SQL plan failed during DuckDB execution. Generate one corrected replacement plan.

Repair rules:
- Treat this as an internal SQL repair. Do not ask the user to rephrase or change the request.
- Preserve the original intent, requested statistics, cohort definitions, and output shape.
- Use the DuckDB execution error to correct SQL syntax, functions, identifiers, or types.
- Return the complete replacement plan, not a patch or an explanation.

input_table_name:
{table_name}

original_user_intent:
{user_intent}

dataset_summary:
{data_summary}

failed_sql_statements:
{failed_statements}

duckdb_execution_error:
{execution_error}
""".strip()
