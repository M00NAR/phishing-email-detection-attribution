import re
import pandas as pd

SRC = "/root/.claude/uploads/68f35678-ae4c-5abf-a779-d10b92f77209/6a5cfb07-fradulent_emails.txt"

with open(SRC, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# split on the mbox "From r " separator line (start of each message)
chunks = re.split(r'\n(?=From r  )', text)
chunks = [c for c in chunks if c.strip()]

rows = []
for i, chunk in enumerate(chunks):
    # headers end at first blank line
    parts = chunk.split("\n\n", 1)
    header_block = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    def grab(field):
        m = re.search(rf'^{field}:\s*(.*)$', header_block, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    subject = grab("Subject")
    from_ = grab("From")
    date = grab("Date")
    rows.append({"idx": i, "from": from_, "subject": subject, "date": date, "body": body.strip()})

df = pd.DataFrame(rows)
df["word_count"] = df["body"].apply(lambda s: len(str(s).split()))

print(f"Total parsed chunks: {len(df)}")
print(f"Empty/near-empty bodies (<5 words): {(df['word_count'] < 5).sum()}")
print(f"Missing subject: {(df['subject']=='').sum()}")
print(f"Missing from: {(df['from']=='').sum()}")
print(f"Duplicate bodies (exact): {df['body'].duplicated().sum()}")
print()
print("Word count distribution:")
print(df["word_count"].describe())
print()
print("Sample subjects:")
print(df["subject"].head(10).to_string())

df.to_csv("/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean/fraud_corpus_parsed.csv", index=False)
