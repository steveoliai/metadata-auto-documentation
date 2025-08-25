# Metadata Auto-Doc v0.3
![Python](https://img.shields.io/badge/Python-3.9+-blue)

*Auto-generate database documentation, ERDs, and schema diffs.*

---

## Overview
The **Metadata Auto-Doc Tool** creates living documentation directly from your database or warehouse, supporting **PostgreSQL** and **BigQuery**.

**Key Features:**
- Extracts schema metadata with columns, relationships, and statistics
- Outputs documentation in **Markdown** and **HTML** formats
- Generates **Mermaid ER diagrams** with foreign key relationships
- Tracks **schema changes** between runs with configurable thresholds
- Stores **timestamped JSON snapshots** for historical analysis
- Optional **Slack alerts** for significant changes

**v0.3 Enhancements:**
- HTML output with styled tables
- Auto-baseline selection from previous snapshots
- Enhanced change tracking with customizable thresholds

---

## Installation
```bash
git clone https://github.com/your-org/metadata-autodoc.git
cd metadata-autodoc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### PostgreSQL
```bash
python metadata_autodoc.py --source postgres \
  --conn postgresql+psycopg2://user:pass@host:5432/db \
  --schema public \
  --md docs.md --html docs.html --json snapshot.json \
  --snapshot-dir ./_snapshots
```

### BigQuery
```bash
# Ensure GOOGLE_APPLICATION_CREDENTIALS is set
python metadata_autodoc.py --source bigquery \
  --project my-project --dataset my_dataset \
  --md docs.md --html docs.html --json snapshot.json \
  --snapshot-dir ./_snapshots
```

---

## Configuration Options

### Data Sources
- `--source postgres` requires `--conn` and `--schema`
- `--source bigquery` requires `--project` and `--dataset`

### Output Formats
- `--md <path>` → Markdown documentation
- `--html <path>` → Styled HTML documentation  
- `--json <path>` → Current snapshot JSON
- `--snapshot-dir <dir>` → Timestamped snapshots for change tracking

### Change Detection
- `--baseline <path>` → Specific snapshot file to compare against
- If omitted with `--snapshot-dir`, automatically uses the most recent snapshot

### Thresholds (defaults shown)
```bash
--threshold-row-pct 10        # Flag row count changes > 10% (Postgres)
--threshold-size-pct 10       # Flag table size changes > 10% (Postgres) 
--threshold-col-desc          # Flag column description changes
```

### Notifications
- `--slack-webhook <url>` → Send change summary to Slack

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