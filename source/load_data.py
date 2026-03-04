# loading the data from the Canada Open Data catalog

import json
import pandas as pd
import os
import re 

data_path = "D:/Harshita Ajmani/Code_harshu/NLP/data/od-do-canada.jsonl"
print("Loading dataset")

records = []

with open(data_path, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"✅ Loaded {len(records):,} records")

# ── Helper functions ──────────────────────────────────────────────────────────

def safe_get_translated(record, field, lang):
    """Extract from bilingual fields like title_translated, notes_translated."""
    val = record.get(field, {})
    if isinstance(val, dict):
        return val.get(lang, "").strip()
    return ""

def clean_text(text):
    """Remove markdown links and extra whitespace."""
    text = re.sub(r'\*\*\[.*?\]\(.*?\)\*\*', '', text)  # remove **[text](url)**
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)            # remove [text](url)
    text = re.sub(r'\s+', ' ', text)                      # collapse whitespace
    return text.strip()

# ── Extract fields ────────────────────────────────────────────────────────────
rows = []
for r in records:
    title_en = safe_get_translated(r, "title_translated", "en") or r.get("title", "")
    title_fr = safe_get_translated(r, "title_translated", "fr") or r.get("title", "")
    desc_en  = clean_text(safe_get_translated(r, "notes_translated", "en"))
    desc_fr  = clean_text(safe_get_translated(r, "notes_translated", "fr"))

    rows.append({
        "id":       r.get("id", ""),
        "title_en": title_en,
        "title_fr": title_fr,
        "desc_en":  desc_en,
        "desc_fr":  desc_fr,
        "subject":  ", ".join(r.get("subject", [])) if isinstance(r.get("subject"), list) else "",
        "org":      r.get("organization", {}).get("title", "") if isinstance(r.get("organization"), dict) else "",
    })

df = pd.DataFrame(rows)
print(f"📊 DataFrame shape: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df[~((df["desc_en"] == "") & (df["desc_fr"] == ""))]
df["desc_en"] = df.apply(lambda x: x["desc_en"] if x["desc_en"] else x["title_en"], axis=1)
df["desc_fr"] = df.apply(lambda x: x["desc_fr"] if x["desc_fr"] else x["title_fr"], axis=1)
df = df.reset_index(drop=True)

print(f"✅ After cleaning: {len(df):,} records")

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"\n📈 Quick stats:")
print(f"  Records with English description : {(df['desc_en'] != '').sum():,}")
print(f"  Records with French description  : {(df['desc_fr'] != '').sum():,}")
print(f"  Records with BOTH languages      : {((df['desc_en'] != '') & (df['desc_fr'] != '')).sum():,}")

print(f"\n📝 Sample record:")
print(f"  ID       : {df.iloc[0]['id']}")
print(f"  Title EN : {df.iloc[0]['title_en']}")
print(f"  Title FR : {df.iloc[0]['title_fr']}")
print(f"  Desc EN  : {df.iloc[0]['desc_en'][:120]}...")
print(f"  Desc FR  : {df.iloc[0]['desc_fr'][:120]}...")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("D:/Harshita Ajmani/Code_harshu/NLP/data/cleaned_data.csv", index=False)
print(f"\n💾 Saved to data/cleaned_data.csv")