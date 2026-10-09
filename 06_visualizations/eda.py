import pandas as pd, numpy as np, re, json, email
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

UP = "/root/.claude/uploads/68f35678-ae4c-5abf-a779-d10b92f77209"
NAZ = f"{UP}/c05bddcb-1791215605517_Nazario_5.csv"
ENR = f"{UP}/3c0650ed-1791215605517_emails_sample_5000.csv"
TASKD = f"{UP}/30576a09-1791215605516_task_d_aggregated.csv"
OUT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/eda"

# palette (dataviz skill reference palette)
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA="#1baf7a"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# ---------- Load Nazario_5.csv ----------
naz = pd.read_csv(NAZ)
naz["label_name"] = naz["label"].map({0: "Legitimate", 1: "Phishing"})
naz["word_count"] = naz["body"].fillna("").apply(lambda s: len(str(s).split()))
naz["char_count"] = naz["body"].fillna("").apply(lambda s: len(str(s)))
naz["subject_len"] = naz["subject"].fillna("").apply(lambda s: len(str(s).split()))

def domain(s):
    if pd.isna(s): return None
    m = re.search(r'@([\w\.-]+)', str(s))
    return m.group(1).lower() if m else None
naz["sender_domain"] = naz["sender"].apply(domain)

urls_is_list = naz["urls"].astype(str).str.startswith("[")
naz["urls_schema"] = np.where(urls_is_list, "list", "flag(0/1)")

summary = {}
summary["nazario_shape"] = naz.shape
summary["nazario_class_balance"] = naz["label_name"].value_counts().to_dict()
summary["nazario_missing"] = naz.isna().sum().to_dict()
summary["nazario_urls_schema_by_class"] = (
    naz.groupby(["label_name", "urls_schema"]).size().unstack(fill_value=0).to_dict()
)
summary["nazario_word_count_by_class"] = naz.groupby("label_name")["word_count"].describe().to_dict()
summary["nazario_top_domains_legit"] = naz[naz.label==0]["sender_domain"].value_counts().head(10).to_dict()
summary["nazario_top_domains_phish"] = naz[naz.label==1]["sender_domain"].value_counts().head(10).to_dict()
# enron.com share within "legitimate"
legit = naz[naz.label==0]
summary["nazario_legit_enron_share"] = float((legit["sender_domain"]=="enron.com").mean())

# ---------- Load emails_sample_5000.csv (raw Enron) ----------
enr = pd.read_csv(ENR)

def parse_body(raw):
    try:
        msg = email.message_from_string(str(raw))
        if msg.is_multipart():
            parts = []
            for p in msg.walk():
                if p.get_content_type() == "text/plain":
                    parts.append(p.get_payload(decode=False) or "")
            return "\n".join(parts)
        else:
            return msg.get_payload(decode=False) or ""
    except Exception:
        return ""

enr["body_parsed"] = enr["message"].apply(parse_body)
enr["word_count"] = enr["body_parsed"].apply(lambda s: len(str(s).split()))
enr["from_"] = enr["message"].apply(lambda m: re.search(r"^From:\s*(.+)$", str(m), re.M).group(1).strip() if re.search(r"^From:\s*(.+)$", str(m), re.M) else None)
enr["message_id"] = enr["message"].apply(lambda m: re.search(r"Message-ID:\s*<(.+?)>", str(m)).group(1) if re.search(r"Message-ID:\s*<(.+?)>", str(m)) else None)

summary["enron_shape"] = enr.shape
summary["enron_word_count_desc"] = enr["word_count"].describe().to_dict()
summary["enron_missing_body"] = int((enr["word_count"]==0).sum())

# overlap check between enron sample and nazario legit (by message content hash / message-id)
naz_legit_ids = set()
# Nazario body sometimes contains "Message-ID:" lines (raw mbox dumped into body) - check
naz["msg_id_in_body"] = naz["body"].fillna("").apply(lambda s: re.search(r"Message-ID:\s*<(.+?)>", s).group(1) if re.search(r"Message-ID:\s*<(.+?)>", s) else None)
naz_ids = set(naz.loc[naz.label==0, "msg_id_in_body"].dropna())
enr_ids = set(enr["message_id"].dropna())
summary["overlap_message_ids"] = len(naz_ids & enr_ids)
summary["naz_legit_with_msgid"] = len(naz_ids)
summary["enron_with_msgid"] = len(enr_ids)

# ---------- task_d_aggregated.csv ----------
taskd = pd.read_csv(TASKD)
summary["task_d"] = taskd.to_dict(orient="records")

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
