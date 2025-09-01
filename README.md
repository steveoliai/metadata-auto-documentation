# Metadata Auto-Doc v0.4

Metadata Auto-Doc is a Python utility that automatically documents database schemas (Postgres, BigQuery), generates ER diagrams, and tracks schema changes across runs.

## Features
- **Supports Postgres and BigQuery**
- **Outputs**: Markdown, HTML (with Mermaid ERD), JSON snapshots
- **Schema Diffing**: Highlights added/removed/changed tables and columns
- **Snapshot Management**: Compare with previous runs to detect changes
- **Slack Notifications**: Optional webhook integration
- **Filters**: Include/exclude tables by regex
- **CI/CD Ready**: Fail with non-zero exit code on significant diffs

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Postgres Example

```bash
python metadata_autodoc.py --source postgres   --conn postgresql+psycopg2://user:pass@host:5432/db   --schema public   --md docs.md --html docs.html --json snapshot.json   --snapshot-dir ./_snapshots
```

### BigQuery Example

```bash
python metadata_autodoc.py --source bigquery   --project my-proj --dataset my_ds   --md docs.md --html docs.html --json snapshot.json   --snapshot-dir ./_snapshots
```

## Options

| Option | Description |
|--------|-------------|
| `--source {postgres,bigquery}` | Required: choose database type |
| `--conn` | SQLAlchemy connection string (Postgres only) |
| `--schema` | Schema name (Postgres only) |
| `--project` | BigQuery project (BigQuery only) |
| `--dataset` | BigQuery dataset (BigQuery only) |
| `--md` | Path to write Markdown documentation |
| `--html` | Path to write HTML documentation |
| `--json` | Path to write current snapshot JSON |
| `--snapshot-dir` | Directory to store timestamped snapshots |
| `--baseline` | Path to baseline JSON (overrides auto-pick) |
| `--threshold-row-pct` | Row count change threshold (%) [default=10] |
| `--threshold-size-pct` | Size change threshold (%) [default=10] |
| `--threshold-col-desc` | Include column description/comment diffs |
| `--include-tables` | Regex filter for table names to include |
| `--exclude-tables` | Regex filter for table names to exclude |
| `--no-erd` | Skip Mermaid ERD block |
| `--slack-webhook` | Slack webhook URL for summary notification |
| `--fail-on-significant` | Exit with non-zero status if significant changes detected |

## Snapshots & Diffs

- Each run can write a timestamped snapshot (`--snapshot-dir`).
- If snapshots exist, the most recent prior run is used as baseline unless `--baseline` is provided.
- Diffs are included in Markdown/HTML outputs and can trigger Slack notifications.

## Slack Notifications

If `--slack-webhook` is provided, a short summary of schema changes is posted. Significant diffs are marked with 🚨.

## Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Run against your database (see examples above)
3. Open the generated `docs.html` in a browser to see the ERD and schema docs

---



## Advanced Example
```bash
python metadata_autodoc.py --source postgres \
  --conn postgresql+psycopg2://user:pass@host:5432/db \
  --schema public \
  --md docs.md --html docs.html --json docs.json \
  --snapshot-dir ./_snapshots --threshold-row-pct 15 \
  --slack-webhook https://hooks.slack.com/services/XXX/YYY/ZZZ
```

---

## Output Examples

### Change Detection Report
```markdown
## Changes vs Baseline
- Tables added: `sessions`
- Tables removed: `temp_staging`
- **users**
  - Columns added: `last_login`
  - Column changed: `email` - data_type: VARCHAR → TEXT
  - Rows: 100,000 → 150,000 (+50%) ⚠️
```

### Mermaid ERD (PostgreSQL)
```mermaid
erDiagram
  users {
    INT id
    TEXT email
  }
  orders {
    INT id
    INT user_id
    FLOAT total
  }
  users ||--o{ orders : FK
```

**Note:** PostgreSQL generates complete ER diagrams with foreign key relationships (`||--o{`), while BigQuery renders tables with columns only.

---

## Database-Specific Features

### PostgreSQL
- Primary keys, foreign keys, and indexes
- Row counts and table sizes
- FK relationships in Mermaid diagrams

### BigQuery  
- Partitioning and clustering information
- Row counts and dataset statistics
- Table-only Mermaid diagrams (no FK relationships)