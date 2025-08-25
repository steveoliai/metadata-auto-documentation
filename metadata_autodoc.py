#!/usr/bin/env python3
"""
Metadata Auto-Doc v0.3
- HTML output
- Mermaid ERD
- Change tracking across runs with snapshots
- Optional Slack summary for significant diffs

Usage examples:
  Postgres:
    python metadata_autodoc.py --source postgres \
      --conn postgresql+psycopg2://user:pass@host:5432/db \
      --schema public \
      --md docs.md --html docs.html --json snapshot.json \
      --snapshot-dir ./_snapshots

  BigQuery:
    python metadata_autodoc.py --source bigquery \
      --project my-proj --dataset my_ds \
      --md docs.md --html docs.html --json snapshot.json \
      --snapshot-dir ./_snapshots
"""
import argparse, os, sys, datetime, json, math, glob
from typing import Any, Dict, List, Optional, Tuple

def _maybe_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

pandas = _maybe_import("pandas")
pd = pandas
sqlalchemy = _maybe_import("sqlalchemy")
bigquery_mod = _maybe_import("google.cloud.bigquery")
jinja2 = _maybe_import("jinja2")
requests = _maybe_import("requests")

# ---------------------- Helpers ----------------------
def human_bytes(n: Optional[int]) -> str:
    if n is None: return "n/a"
    units = ["B","KB","MB","GB","TB","PB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}"

def pct_change(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base is None: return None
    if base == 0: return None
    return ((cur - base) / base) * 100.0

# ---------------------- Collectors ----------------------
def pg_collect(conn: str, schema: str) -> Dict[str, Any]:
    if sqlalchemy is None or pd is None:
        raise RuntimeError("SQLAlchemy and pandas required for PostgreSQL.")
    eng = sqlalchemy.create_engine(conn)
    info: Dict[str, Any] = {"flavor": "postgres", "schema": schema, "tables": {}}

    cols_sql = """
    SELECT c.table_name, c.column_name, c.data_type, c.is_nullable,
           c.column_default
    FROM information_schema.columns c
    WHERE c.table_schema = %(schema)s
    ORDER BY c.table_name, c.ordinal_position
    """
    cols = pd.read_sql_query(cols_sql, eng, params={"schema": schema})

    tbl_comments_sql = """
    SELECT c.relname AS table_name, d.description AS comment
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
    WHERE n.nspname = %(schema)s AND c.relkind = 'r'
    """
    tbl_comments = pd.read_sql_query(tbl_comments_sql, eng, params={"schema": schema})

    col_comments_sql = """
    SELECT c.relname AS table_name, a.attname AS column_name, d.description AS comment
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
    LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = a.attnum
    WHERE n.nspname = %(schema)s AND c.relkind = 'r'
    """
    col_comments = pd.read_sql_query(col_comments_sql, eng, params={"schema": schema})

    pk_sql = """
    SELECT k.table_name, k.column_name
    FROM information_schema.table_constraints t
    JOIN information_schema.key_column_usage k
      ON t.constraint_name = k.constraint_name
     AND t.table_schema = k.table_schema
    WHERE t.table_schema = %(schema)s AND t.constraint_type = 'PRIMARY KEY'
    ORDER BY k.table_name, k.ordinal_position
    """
    pks = pd.read_sql_query(pk_sql, eng, params={"schema": schema})

    fk_sql = """
    SELECT tc.table_name, kcu.column_name,
           ccu.table_name AS foreign_table_name,
           ccu.column_name AS foreign_column_name
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema = tc.table_schema
    WHERE tc.table_schema = %(schema)s AND tc.constraint_type = 'FOREIGN KEY'
    ORDER BY tc.table_name, kcu.ordinal_position
    """
    fks = pd.read_sql_query(fk_sql, eng, params={"schema": schema})

    idx_sql = """
    SELECT schemaname AS schema_name, tablename AS table_name, indexname, indexdef
    FROM pg_indexes WHERE schemaname = %(schema)s
    """
    idx = pd.read_sql_query(idx_sql, eng, params={"schema": schema})

    rowcount_sql = """
    SELECT relname AS table_name, n_live_tup AS approx_rows
    FROM pg_stat_user_tables WHERE schemaname = %(schema)s
    """
    rowcounts = pd.read_sql_query(rowcount_sql, eng, params={"schema": schema})

    size_rows = []
    for t in rowcounts["table_name"].tolist():
        size_sql = f"SELECT pg_total_relation_size('{schema}.{t}') AS size_bytes"
        s = pd.read_sql_query(size_sql, eng).iloc[0,0]
        size_rows.append({"table_name": t, "size_bytes": int(s) if s is not None else None})
    sizes = pd.DataFrame(size_rows)

    eng.dispose()

    for tname in sorted(cols["table_name"].unique()):
        tcols = cols[cols["table_name"] == tname]
        tcol_comments = col_comments[col_comments["table_name"] == tname]
        tpk = pks[pks["table_name"] == tname]["column_name"].tolist()
        tfk = fks[fks["table_name"] == tname][["column_name","foreign_table_name","foreign_column_name"]]
        tidx = idx[idx["table_name"] == tname][["indexname","indexdef"]]
        trow = rowcounts[rowcounts["table_name"] == tname]
        tsize = sizes[sizes["table_name"] == tname]
        tcomment = tbl_comments[tbl_comments["table_name"] == tname]["comment"]
        info["tables"][tname] = {
            "comment": (tcomment.iloc[0] if not tcomment.empty else None),
            "approx_rows": int(trow["approx_rows"].iloc[0]) if not trow.empty else None,
            "size_bytes": int(tsize["size_bytes"].iloc[0]) if not tsize.empty else None,
            "primary_key": tpk,
            "foreign_keys": tfk.to_dict(orient="records"),
            "indexes": tidx.to_dict(orient="records"),
            "columns": []
        }
        for _, r in tcols.iterrows():
            cmt = tcol_comments[tcol_comments["column_name"] == r["column_name"]]["comment"]
            info["tables"][tname]["columns"].append({
                "name": r["column_name"],
                "data_type": r["data_type"],
                "nullable": (r["is_nullable"] == "YES"),
                "default": r["column_default"],
                "comment": (cmt.iloc[0] if not cmt.empty else None)
            })
    return info

def bq_collect(project: str, dataset: str) -> Dict[str, Any]:
    if bigquery_mod is None:
        raise RuntimeError("google-cloud-bigquery required for BigQuery.")
    client = bigquery_mod.Client(project=project)
    info: Dict[str, Any] = {"flavor": "bigquery", "project": project, "dataset": dataset, "tables": {}}
    for t in client.list_tables(f"{project}.{dataset}"):
        table = client.get_table(t)
        cols = []
        for f in table.schema:
            cols.append({
                "name": f.name,
                "data_type": f.field_type,
                "mode": f.mode,
                "description": f.description
            })
        info["tables"][t.table_id] = {
            "description": table.description,
            "partitioning": str(table.time_partitioning.type_) if table.time_partitioning else None,
            "partition_field": table.time_partitioning.field if table.time_partitioning else None,
            "clustering_fields": list(table.clustering_fields) if table.clustering_fields else [],
            "num_rows": int(table.num_rows),
            "table_type": table.table_type,
            "columns": cols
        }
    return info

# ---------------------- ERD (Mermaid) ----------------------
def _map_dtype(dtype: str) -> str:
    d = (dtype or "").lower()
    if "int" in d: return "INT"
    if "float" in d or "double" in d or "numeric" in d or "number" in d: return "FLOAT"
    if "timestamp" in d or "date" in d or "time" in d: return "TIMESTAMP"
    if "bool" in d: return "BOOLEAN"
    return "TEXT"

def mermaid_from_pg(meta: Dict[str, Any]) -> str:
    lines = ["erDiagram"]
    for tname, tinfo in meta["tables"].items():
        lines.append(f"  {tname} {{")
        for c in tinfo["columns"]:
            lines.append(f"    {_map_dtype(c.get('data_type',''))} {c['name']}")
        lines.append("  }")
    # relationships
    for tname, tinfo in meta["tables"].items():
        for fk in tinfo.get("foreign_keys", []):
            ft = fk["foreign_table_name"]
            if ft in meta["tables"]:
                # simple cardinality
                lines.append(f"  {ft} ||--o{{ {tname} : FK")
    return "\n".join(lines)

def mermaid_from_bq(meta: Dict[str, Any]) -> str:
    lines = ["erDiagram"]
    for tname, tinfo in meta["tables"].items():
        lines.append(f"  {tname} {{")
        for c in tinfo["columns"]:
            lines.append(f"    {_map_dtype(c.get('data_type',''))} {c['name']}")
        lines.append("  }")
    return "\n".join(lines)

# ---------------------- Diff ----------------------
def diff_meta(cur: Dict[str, Any], base: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "tables_added": [], "tables_removed": [], "tables_changed": [],
        "summary": {"significant": False}
    }
    cur_tables = set(cur["tables"].keys())
    base_tables = set(base["tables"].keys())
    out["tables_added"] = sorted(list(cur_tables - base_tables))
    out["tables_removed"] = sorted(list(base_tables - cur_tables))

    # Table-by-table comparison
    for t in sorted(cur_tables & base_tables):
        cti = cur["tables"][t]
        bti = base["tables"][t]
        changed = {"table": t, "columns_added": [], "columns_removed": [], "columns_changed": [], "keys_changed": False, "options_changed": [], "row_size_change": {}}

        # Columns
        cur_cols = {c["name"]: c for c in cti["columns"]}
        base_cols = {c["name"]: c for c in bti["columns"]}
        for name in cur_cols.keys() - base_cols.keys():
            changed["columns_added"].append(name)
        for name in base_cols.keys() - cur_cols.keys():
            changed["columns_removed"].append(name)
        for name in cur_cols.keys() & base_cols.keys():
            cc, bc = cur_cols[name], base_cols[name]
            # different attributes
            diffs = {}
            for k in ["data_type", "nullable", "default", "comment", "mode", "description"]:
                if cc.get(k) != bc.get(k):
                    diffs[k] = {"baseline": bc.get(k), "current": cc.get(k)}
            if diffs:
                changed["columns_changed"].append({"name": name, "diffs": diffs})

        # Keys / Indexes / Partitioning/Clustering
        flavor = cur.get("flavor")
        if flavor == "postgres":
            if cti.get("primary_key") != bti.get("primary_key"): changed["keys_changed"] = True
            if cti.get("foreign_keys") != bti.get("foreign_keys"): changed["keys_changed"] = True
            if cti.get("indexes") != bti.get("indexes"): changed["keys_changed"] = True
            # rowcount/size
            rc_cur, rc_base = cti.get("approx_rows"), bti.get("approx_rows")
            sz_cur, sz_base = cti.get("size_bytes"), bti.get("size_bytes")
            row_pct = pct_change(rc_cur, rc_base)
            size_pct = pct_change(sz_cur, sz_base)
            thr_row = thresholds.get("row_pct", 10.0)
            thr_size = thresholds.get("size_pct", 10.0)
            changed["row_size_change"] = {
                "row_count_baseline": rc_base, "row_count_current": rc_cur, "row_count_delta_pct": row_pct, "row_count_flag": (abs(row_pct) > thr_row) if row_pct is not None else False,
                "size_baseline": sz_base, "size_current": sz_cur, "size_delta_pct": size_pct, "size_flag": (abs(size_pct) > thr_size) if size_pct is not None else False
            }
            if changed["row_size_change"]["row_count_flag"] or changed["row_size_change"]["size_flag"]:
                out["summary"]["significant"] = True
        else:
            # BigQuery table options
            if cti.get("partitioning") != bti.get("partitioning"): changed["options_changed"].append("partitioning")
            if cti.get("partition_field") != bti.get("partition_field"): changed["options_changed"].append("partition_field")
            if (cti.get("clustering_fields") or []) != (bti.get("clustering_fields") or []): changed["options_changed"].append("clustering_fields")
            if cti.get("num_rows") != bti.get("num_rows"):
                changed["row_size_change"] = {"num_rows_baseline": bti.get("num_rows"), "num_rows_current": cti.get("num_rows"), "row_count_delta_pct": pct_change(cti.get("num_rows"), bti.get("num_rows"))}
        if changed["columns_added"] or changed["columns_removed"] or changed["columns_changed"] or changed["keys_changed"] or changed["options_changed"] or changed["row_size_change"]:
            out["tables_changed"].append(changed)
            if changed["columns_added"] or changed["columns_removed"] or changed["columns_changed"] or changed["keys_changed"]:
                out["summary"]["significant"] = True
    return out

# ---------------------- Render ----------------------
def render_markdown(meta: Dict[str, Any], erd_block: str, diffs: Optional[Dict[str, Any]]) -> str:
    now = datetime.datetime.utcnow().isoformat() + "Z"
    title = "PostgreSQL Schema" if meta.get("flavor") == "postgres" else "BigQuery Dataset"
    lines = []
    lines.append(f"# Metadata Documentation - {title}")
    if meta.get("flavor") == "postgres":
        lines.append(f"**Schema**: `{meta['schema']}`  ")
    else:
        lines.append(f"**Project**: `{meta['project']}`  \n**Dataset**: `{meta['dataset']}`  ")
    lines.append(f"_Generated {now}_\n")
    if erd_block:
        lines.append("## ER Diagram")
        lines.append("```mermaid")
        lines.append(erd_block)
        lines.append("```")

    if diffs:
        lines.append("\n## Changes vs Baseline")
        if diffs["tables_added"]:
            lines.append(f"- Tables added: `{', '.join(diffs['tables_added'])}`")
        if diffs["tables_removed"]:
            lines.append(f"- Tables removed: `{', '.join(diffs['tables_removed'])}`")
        if diffs["tables_changed"]:
            lines.append("\n### Table-level changes")
            for ch in diffs["tables_changed"]:
                lines.append(f"- **{ch['table']}**")
                if ch["columns_added"]:
                    lines.append(f"  - Columns added: `{', '.join(ch['columns_added'])}`")
                if ch["columns_removed"]:
                    lines.append(f"  - Columns removed: `{', '.join(ch['columns_removed'])}`")
                for cc in ch["columns_changed"]:
                    diffs_str = ", ".join([f"{k}: {v['baseline']} → {v['current']}" for k, v in cc["diffs"].items()])
                    lines.append(f"  - Column changed: `{cc['name']}` - {diffs_str}")
                if ch["keys_changed"]:
                    lines.append("  - Keys/indexes changed")
                if ch["options_changed"]:
                    lines.append(f"  - Table options changed: {', '.join(ch['options_changed'])}")
                if ch.get("row_size_change"):
                    rsc = ch["row_size_change"]
                    if meta.get("flavor") == "postgres":
                        lines.append(f"  - Rows: {rsc.get('row_count_baseline')} → {rsc.get('row_count_current')} ({(rsc.get('row_count_delta_pct') or 0):.2f}%){' ⚠️' if rsc.get('row_count_flag') else ''}")
                        lines.append(f"  - Size: {human_bytes(rsc.get('size_baseline'))} → {human_bytes(rsc.get('size_current'))} ({(rsc.get('size_delta_pct') or 0):.2f}%){' ⚠️' if rsc.get('size_flag') else ''}")
                    else:
                        lines.append(f"  - Rows: {rsc.get('num_rows_baseline')} → {rsc.get('num_rows_current')} ({(rsc.get('row_count_delta_pct') or 0):.2f}%)")
        else:
            lines.append("_No changes detected._")

    # Tables
    for tname, tinfo in sorted(meta["tables"].items()):
        lines.append(f"\n## {tname}")
        if meta.get("flavor") == "postgres":
            if tinfo.get("comment"):
                lines.append(f"> {tinfo['comment']}")
            lines.append(f"- Approx rows: **{tinfo.get('approx_rows','n/a')}**")
            lines.append(f"- Size: **{human_bytes(tinfo.get('size_bytes'))}**")
            if tinfo.get("primary_key"):
                lines.append(f"- Primary key: `{', '.join(tinfo['primary_key'])}`")
            if tinfo.get("foreign_keys"):
                lines.append(f"- Foreign keys:")
                for fk in tinfo["foreign_keys"]:
                    lines.append(f"  - `{fk['column_name']}` → `{fk['foreign_table_name']}.{fk['foreign_column_name']}`")
            if tinfo.get("indexes"):
                lines.append(f"- Indexes:")
                for idx in tinfo["indexes"]:
                    lines.append(f"  - `{idx['indexname']}` - `{idx['indexdef']}`")
            lines.append("\n### Columns")
            lines.append("| name | data_type | nullable | default | comment |")
            lines.append("|---|---|---|---|---|")
            for c in tinfo["columns"]:
                lines.append(f"| {c['name']} | {c['data_type']} | {c['nullable']} | {str(c['default']) if c['default'] is not None else ''} | {c.get('comment','') or ''} |")
        else:
            if tinfo.get("description"):
                lines.append(f"> {tinfo['description']}")
            lines.append(f"- Table type: **{tinfo.get('table_type')}**")
            lines.append(f"- Rows: **{tinfo.get('num_rows','n/a')}**")
            if tinfo.get("partitioning"):
                lines.append(f"- Partitioning: `{tinfo['partitioning']}` on `{tinfo.get('partition_field')}`")
            if tinfo.get("clustering_fields"):
                lines.append(f"- Clustering: `{', '.join(tinfo['clustering_fields'])}`")
            lines.append("\n### Columns")
            lines.append("| name | data_type | mode | description |")
            lines.append("|---|---|---|---|")
            for c in tinfo["columns"]:
                lines.append(f"| {c['name']} | {c['data_type']} | {c.get('mode','')} | {c.get('description','') or ''} |")
    return "\n".join(lines)

def render_html(meta, erd_block, diffs):
    # Ensure Mermaid renders by embedding the script and using <div class="mermaid">.
    if jinja2 is None:
        erd_html = f'<div class="mermaid">{erd_block}</div>' if erd_block else ""
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Metadata Auto-Doc v0.3.1</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f6f6f6; }}
.small {{ color: #666; font-size: 12px; }}
.mermaid {{ margin: 12px 0; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true }});</script>
</head>
<body>
<h1>Metadata Auto-Doc <span class="small">(v0.3.1)</span></h1>
{erd_html}
</body></html>"""

    tpl = jinja2.Template("""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Metadata Auto-Doc v0.3.1</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
th { background: #f6f6f6; }
.small { color: #666; font-size: 12px; }
code { background: #f7f7f7; padding: 2px 4px; border-radius: 4px; }
.mermaid { margin: 12px 0; }
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({ startOnLoad: true });</script>
</head>
<body>
<h1>Metadata Auto-Doc <span class="small">(v0.3.1)</span></h1>
<p>
  {% if meta.flavor == 'postgres' %}
    <strong>Schema:</strong> {{ meta.schema }}
  {% else %}
    <strong>Project:</strong> {{ meta.project }} &nbsp; <strong>Dataset:</strong> {{ meta.dataset }}
  {% endif %}
</p>

{% if erd_block %}
<h2>ER Diagram</h2>
<div class="mermaid">{{ erd_block }}</div>
{% endif %}

{% if diffs %}
<h2>Changes vs Baseline</h2>
{% if diffs.tables_added %}<p>Tables added: <code>{{ diffs.tables_added|join(', ') }}</code></p>{% endif %}
{% if diffs.tables_removed %}<p>Tables removed: <code>{{ diffs.tables_removed|join(', ') }}</code></p>{% endif %}

{% if diffs.tables_changed %}
<h3>Table-level changes</h3>
<ul>
{% for ch in diffs.tables_changed %}
  <li>
    <strong>{{ ch.table }}</strong>
    <ul>
      {% if ch.columns_added %}<li>Columns added: <code>{{ ch.columns_added|join(', ') }}</code></li>{% endif %}
      {% if ch.columns_removed %}<li>Columns removed: <code>{{ ch.columns_removed|join(', ') }}</code></li>{% endif %}
      {% for cc in ch.columns_changed %}
        <li>Column changed: <code>{{ cc.name }}</code> -
          {% for k,v in cc.diffs.items() %}<code>{{ k }}</code>: {{ v.baseline }} → {{ v.current }}{% if not loop.last %}, {% endif %}{% endfor %}
        </li>
      {% endfor %}
      {% if ch.keys_changed %}<li>Keys/indexes changed</li>{% endif %}
      {% if ch.options_changed %}<li>Options changed: <code>{{ ch.options_changed|join(', ') }}</code></li>{% endif %}
      {% if ch.row_size_change %}
        {% if meta.flavor == 'postgres' %}
          <li>Rows: {{ ch.row_size_change.row_count_baseline }} → {{ ch.row_size_change.row_count_current }} ({{ '%.2f'|format(ch.row_size_change.row_count_delta_pct or 0) }}%)</li>
          <li>Size: {{ ch.row_size_change.size_baseline|default('') }} → {{ ch.row_size_change.size_current|default('') }}</li>
        {% else %}
          <li>Rows: {{ ch.row_size_change.num_rows_baseline }} → {{ ch.row_size_change.num_rows_current }} ({{ '%.2f'|format(ch.row_size_change.row_count_delta_pct or 0) }}%)</li>
        {% endif %}
      {% endif %}
    </ul>
  </li>
{% endfor %}
</ul>
{% else %}
<p><em>No changes detected.</em></p>
{% endif %}
{% endif %}

<h2>Tables</h2>
{% for tname, tinfo in meta.tables.items() %}
  <h3 id="{{ tname }}">{{ tname }}</h3>
  {% if meta.flavor == 'postgres' %}
    {% if tinfo.comment %}<p>{{ tinfo.comment }}</p>{% endif %}
    <ul>
      <li>Approx rows: <strong>{{ tinfo.approx_rows or 'n/a' }}</strong></li>
      <li>Size: <strong>{{ tinfo.size_bytes or 'n/a' }}</strong></li>
      {% if tinfo.primary_key %}<li>Primary key: <code>{{ tinfo.primary_key|join(', ') }}</code></li>{% endif %}
      {% if tinfo.foreign_keys %}
        <li>Foreign keys:
          <ul>
            {% for fk in tinfo.foreign_keys %}
              <li><code>{{ fk.column_name }}</code> → <code>{{ fk.foreign_table_name }}.{{ fk.foreign_column_name }}</code></li>
            {% endfor %}
          </ul>
        </li>
      {% endif %}
      {% if tinfo.indexes %}
        <li>Indexes:
          <ul>
            {% for idx in tinfo.indexes %}
              <li><code>{{ idx.indexname }}</code> - <code>{{ idx.indexdef }}</code></li>
            {% endfor %}
          </ul>
        </li>
      {% endif %}
    </ul>
    <table>
      <thead><tr><th>name</th><th>data_type</th><th>nullable</th><th>default</th><th>comment</th></tr></thead>
      <tbody>
        {% for c in tinfo.columns %}
        <tr><td>{{ c.name }}</td><td>{{ c.data_type }}</td><td>{{ c.nullable }}</td><td>{{ c.default }}</td><td>{{ c.comment or '' }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    {% if tinfo.description %}<p>{{ tinfo.description }}</p>{% endif %}
    <ul>
      <li>Table type: <strong>{{ tinfo.table_type }}</strong></li>
      <li>Rows: <strong>{{ tinfo.num_rows or 'n/a' }}</strong></li>
      {% if tinfo.partitioning %}<li>Partitioning: <code>{{ tinfo.partitioning }}</code> on <code>{{ tinfo.partition_field }}</code></li>{% endif %}
      {% if tinfo.clustering_fields %}<li>Clustering: <code>{{ tinfo.clustering_fields|join(', ') }}</code></li>{% endif %}
    </ul>
    <table>
      <thead><tr><th>name</th><th>data_type</th><th>mode</th><th>description</th></tr></thead>
      <tbody>
        {% for c in tinfo.columns %}
        <tr><td>{{ c.name }}</td><td>{{ c.data_type }}</td><td>{{ c.mode }}</td><td>{{ c.description or '' }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  {% endif %}
{% endfor %}
</body></html>
""")
    return tpl.render(meta=meta, erd_block=erd_block, diffs=diffs)

# ---------------------- Notifications ----------------------
def notify_slack(webhook_url: str, text: str):
    if requests is None:
        print("requests not installed; skipping Slack notification")
        return
    try:
        resp = requests.post(webhook_url, json={"text": text}, timeout=10)
        if resp.status_code >= 300:
            print(f"Slack webhook returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Slack notification failed: {e}")

def summarize_diffs(diffs: Dict[str, Any]) -> str:
    if not diffs: return "No baseline to compare."
    parts = []
    if diffs.get("tables_added"): parts.append(f"Tables added: {', '.join(diffs['tables_added'])}")
    if diffs.get("tables_removed"): parts.append(f"Tables removed: {', '.join(diffs['tables_removed'])}")
    changed = diffs.get("tables_changed") or []
    if changed:
        parts.append(f"Tables changed: {', '.join(ch['table'] for ch in changed)}")
    if not parts:
        return "No changes detected."
    return "\n".join(parts)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["postgres","bigquery"], required=True)
    ap.add_argument("--conn", help="SQLAlchemy connection string for Postgres")
    ap.add_argument("--schema", help="Postgres schema")
    ap.add_argument("--project", help="BigQuery project")
    ap.add_argument("--dataset", help="BigQuery dataset")

    ap.add_argument("--md", help="Output Markdown path")
    ap.add_argument("--html", help="Output HTML path")
    ap.add_argument("--json", help="Write current snapshot JSON to this path")
    ap.add_argument("--snapshot-dir", help="Directory to store timestamped snapshots and auto-pick baseline from")
    ap.add_argument("--baseline", help="Path to baseline snapshot JSON (overrides auto-pick)")

    ap.add_argument("--threshold-row-pct", type=float, default=10.0, help="Flag Postgres row count changes above this percent")
    ap.add_argument("--threshold-size-pct", type=float, default=10.0, help="Flag Postgres size changes above this percent")
    ap.add_argument("--threshold-col-desc", action="store_true", help="Flag column description/comment changes")

    ap.add_argument("--slack-webhook", help="Slack webhook URL for summary notification (optional)")
    args = ap.parse_args()

    if args.source == "postgres":
        if not (args.conn and args.schema):
            raise SystemExit("--conn and --schema are required for Postgres")
        meta = pg_collect(args.conn, args.schema)
    else:
        if not (args.project and args.dataset):
            raise SystemExit("--project and --dataset are required for BigQuery")
        meta = bq_collect(args.project, args.dataset)

    # ERD
    erd_block = mermaid_from_pg(meta) if meta.get("flavor") == "postgres" else mermaid_from_bq(meta)

    # Snapshots
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    current_snapshot = {"generated_at_utc": ts, **meta}

    # Write JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump(current_snapshot, f, indent=2, default=str)

    # Timestamped snapshot dir
    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        ts_path = os.path.join(args.snapshot_dir, f"meta_snapshot_{ts}.json")
        with open(ts_path, "w") as f:
            json.dump(current_snapshot, f, indent=2, default=str)

    # Pick baseline
    baseline = None
    if args.baseline:
        try:
            with open(args.baseline, "r") as f:
                baseline = json.load(f)
        except Exception:
            baseline = None
    elif args.snapshot_dir:
        # pick most recent prior
        patt = os.path.join(args.snapshot_dir, "meta_snapshot_*.json")
        paths = sorted(glob.glob(patt))
        if len(paths) >= 2:
            baseline_path = paths[-2]  # previous one
            try:
                with open(baseline_path, "r") as f:
                    baseline = json.load(f)
            except Exception:
                baseline = None

    # Diff
    diffs = None
    if baseline:
        thr = {"row_pct": args.threshold_row_pct, "size_pct": args.threshold_size_pct, "flag_desc": args.threshold_col_desc}
        diffs = diff_meta(current_snapshot, baseline, thr)

    # Render outputs
    if args.md:
        md = render_markdown(meta, erd_block, diffs)
        with open(args.md, "w") as f:
            f.write(md)

    if args.html:
        html = render_html(meta, erd_block, diffs)
        with open(args.html, "w") as f:
            f.write(html)

    # Slack notify if significant
    if args.slack_webhook and diffs and (diffs.get("summary", {}).get("significant") or diffs.get("tables_added") or diffs.get("tables_removed")):
        notify_slack(args.slack_webhook, "Metadata changes detected:\n" + summarize_diffs(diffs))

    print("Done.")

if __name__ == "__main__":
    main()
