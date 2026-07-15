"""
Advanced Inbox Declutter Copilot
=================================
Streamlit + LangGraph + Gemini API + imaplib
"""

import imaplib
import email
import json
import os
import re
import time
from datetime import date, timedelta
from email.header import decode_header
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

import google.generativeai as genai
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GMAIL_ADDRESS = os.getenv("GMAIL_SENDER_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Inbox Declutter Copilot",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Hero title */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: rgba(200,200,220,0.7);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* VIP card */
    .vip-card {
        background: linear-gradient(135deg, rgba(251,191,36,0.18), rgba(245,158,11,0.08));
        border: 1px solid rgba(251,191,36,0.4);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.7rem;
        transition: transform 0.2s;
    }
    .vip-card:hover { transform: translateY(-2px); }
    .vip-badge {
        display:inline-block;
        background: linear-gradient(90deg,#fbbf24,#f59e0b);
        color:#1a1a1a;
        border-radius:999px;
        padding:2px 10px;
        font-size:0.7rem;
        font-weight:700;
        letter-spacing:0.05em;
        margin-right:8px;
    }

    /* Category header */
    .cat-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a78bfa;
        margin: 0.3rem 0 0.1rem;
    }

    /* Sender pill */
    .sender-pill {
        display:inline-block;
        background:rgba(167,139,250,0.15);
        border:1px solid rgba(167,139,250,0.3);
        border-radius:999px;
        padding:2px 10px;
        font-size:0.78rem;
        margin:2px 3px;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }

    /* Metric boxes */
    .metric-box {
        background:rgba(255,255,255,0.05);
        border:1px solid rgba(255,255,255,0.1);
        border-radius:10px;
        padding:0.7rem 1rem;
        text-align:center;
    }
    .metric-box .metric-val {
        font-size:1.8rem;
        font-weight:700;
        color:#60a5fa;
    }
    .metric-box .metric-label {
        font-size:0.75rem;
        color:rgba(200,200,220,0.6);
    }

    /* Expander tweaks */
    [data-testid="stExpander"] {
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.1);
        border-radius:10px 10px 0 0;
        margin-bottom:0;
    }

    /* Action panel — always visible beneath each category expander */
    .action-panel {
        background: rgba(96,165,250,0.07);
        border: 1px solid rgba(96,165,250,0.25);
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 0.9rem 1.2rem 1rem;
        margin-bottom: 1.2rem;
    }
    .action-panel-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(200,200,220,0.55);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    /* Mini per-sender action row inside expander */
    .sender-mini-action {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.45rem 0.8rem 0.5rem;
        margin-top: 0.3rem;
        margin-bottom: 0.8rem;
    }
    .sender-mini-label {
        font-size:0.72rem;
        color:rgba(200,200,220,0.45);
        letter-spacing:0.05em;
        text-transform:uppercase;
        margin-bottom:0.25rem;
    }
    .vip-action-panel {
        background: rgba(251,191,36,0.07);
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 10px;
        padding: 0.9rem 1.2rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .del-confirm {
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-size: 0.84rem;
        color: #fca5a5;
        margin-bottom: 0.4rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────
class InboxState(TypedDict):
    start_date: str
    end_date: str
    vip_senders: List[str]
    raw_emails: List[Dict[str, Any]]
    sender_groups: Dict[str, Dict[str, Any]]   # {email: {count, subjects, uids, category, is_vip, name}}
    error: Optional[str]
    action: Optional[str]          # "move" | "delete" | None
    action_category: Optional[str]
    action_folder: Optional[str]
    has_attachments: bool
    attachment_size_mb: float
    total_uids_found: int
    is_fallback_search: bool
    categorization_error: Optional[str]
    unread_only: bool


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _decode_header_value(raw: Any) -> str:
    """Safely decode an email header value."""
    if raw is None:
        return ""
    decoded_parts = decode_header(str(raw))
    result = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            try:
                result.append(part.decode(enc or "utf-8", errors="replace"))
            except Exception:
                result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result).strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from a 'Name <email>' string."""
    match = re.search(r"<(.+?)>", raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def _extract_display_name(raw: str) -> str:
    """Extract display name from a 'Name <email>' string."""
    match = re.search(r"^(.*?)\s*<", raw)
    if match:
        name = match.group(1).strip().strip('"').strip("'")
        if name:
            return name
    return raw.split("@")[0].strip()


def _imap_connect() -> imaplib.IMAP4_SSL:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    return mail


def _date_to_imap(d: date) -> str:
    """Convert Python date to IMAP date string (D-Mon-YYYY). No leading zero for day."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{d.day}-{months[d.month-1]}-{d.year}"


def _parse_bodystructure(structure: Any) -> tuple[bool, int]:
    """
    Recursively parse BODYSTRUCTURE to detect attachments and calculate size.
    Returns: (has_attachment, total_size_bytes)
    """
    has_att = False
    size = 0
    
    if isinstance(structure, list):
        # Multipart message
        if any(isinstance(x, list) for x in structure):
            # Nested multiparts
            for part in structure:
                if isinstance(part, list):
                    h, s = _parse_bodystructure(part)
                    has_att = has_att or h
                    size = size + s  # Explicit add to avoid augmented assignment issues if size is unexpected type
        else:
            # Single part within a multipart, or leaf part
            # Check for 'attachment' in disposition (usually index 8 in simple structures)
            # or filename
            part_str = str(structure).lower()
            if 'attachment' in part_str or 'filename' in part_str:
                has_att = True
                # Size is usually at index 6 for leaf parts
                if len(structure) > 6 and isinstance(structure[6], int):
                    size += structure[6]
    return has_att, size


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 – FETCH EMAILS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_emails_node(state: InboxState) -> InboxState:
    """Connect to Gmail via IMAP and fetch unread emails in the date range."""
    try:
        mail = _imap_connect()
        mail.select("INBOX")

        start_str = _date_to_imap(date.fromisoformat(state["start_date"]))
        end_str = _date_to_imap(date.fromisoformat(state["end_date"]) + timedelta(days=1))

        # Search for emails
        # 1. Try with date range
        date_criteria = f"SINCE {start_str} BEFORE {end_str}"
        unread_criteria = "UNSEEN" if state.get("unread_only", True) else "ALL"
        search_query = f"({unread_criteria} {date_criteria})"
        
        status, data = mail.uid("search", None, search_query)
        
        uid_list = data[0].split()
        
        # 2. Fallback: If no results in range and unread_only was on, try ANY UNSEEN to be helpful
        is_fallback_search = False
        if (status != "OK" or not uid_list) and state.get("unread_only", True):
            status, data = mail.uid("search", None, "(UNSEEN)")
            if status == "OK":
                uid_list = data[0].split()
                is_fallback_search = True

        if not uid_list:
            state["raw_emails"] = []
            state["total_uids_found"] = 0
            mail.logout()
            return state

        # Diagnostic: store count of UIDs found
        state["total_uids_found"] = len(uid_list)
        state["is_fallback_search"] = is_fallback_search

        raw_emails = []
        # Fetch in batches of 50 to avoid timeouts
        batch_size = 50
        for i in range(0, len(uid_list), batch_size):
            batch = uid_list[i : i + batch_size]
            uid_str = b",".join(batch).decode()
            # OPTIMIZED: Fetch metadata only
            status2, msg_data = mail.uid("fetch", uid_str, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)] RFC822.SIZE BODYSTRUCTURE UID)")
            if status2 != "OK":
                continue
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # The first element is the metadata (UID, SIZE, etc.)
                    # The second element is the header content
                    metadata_raw = response_part[0].decode()
                    header_content = response_part[1]
                    
                    uid_match = re.search(r"UID\s+(\d+)", metadata_raw, re.IGNORECASE)
                    uid = uid_match.group(1) if uid_match else ""

                    msg = email.message_from_bytes(header_content)
                    subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                    sender_raw = _decode_header_value(msg.get("From", ""))
                    msg_date = _decode_header_value(msg.get("Date", ""))
                    
                    # Extract size and attachments from the metadata raw string
                    size_match = re.search(r"RFC822.SIZE\s+(\d+)", metadata_raw, re.IGNORECASE)
                    rfc822_size = int(size_match.group(1)) if size_match else 0
                    
                    # More robust attachment check in metadata
                    has_att = "attachment" in metadata_raw.lower() or "filename=" in metadata_raw.lower()
                    
                    raw_emails.append({
                        "uid": uid,
                        "subject": subject[:120],
                        "sender_raw": sender_raw,
                        "sender_email": _extract_email_address(sender_raw),
                        "sender_name": _extract_display_name(sender_raw),
                        "date": msg_date,
                        "size_bytes": rfc822_size,
                        "has_attachment": has_att
                    })
        

        state["raw_emails"] = raw_emails
        state["error"] = None
        mail.logout()
    except Exception as exc:
        state["error"] = f"Fetch error: {exc}"
        state["raw_emails"] = []
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 – CATEGORISE WITH GEMINI
# ─────────────────────────────────────────────────────────────────────────────
def categorise_senders_node(state: InboxState) -> InboxState:
    """Group emails by sender, then call Gemini to assign category labels."""
    raw_emails = state.get("raw_emails", [])
    if not raw_emails:
        state["sender_groups"] = {}
        return state

    # Group by sender_email
    groups: Dict[str, Dict] = {}
    for em in raw_emails:
        se = em["sender_email"]
        if se not in groups:
            groups[se] = {
                "name": em["sender_name"],
                "email": se,
                "count": 0,
                "subjects": [],
                "uids": [],
                "category": "Uncategorised",
                "is_vip": False,
            }
        groups[se]["count"] += 1
        groups[se]["uids"].append(em["uid"])
        if len(groups[se]["subjects"]) < 5:
            groups[se]["subjects"].append(em["subject"])

    # Build prompt payload
    sender_list = [
        {"email": v["email"], "name": v["name"], "email_count": v["count"]}
        for v in groups.values()
    ]

    prompt = f"""You are an expert email classifier. 
Assign ONE category to each sender. Use the most specific category. 
CATEGORIES: Newsletters, Promotions, Social Media, Internal Updates, Finance & Billing, Travel & Bookings, Spam, Shopping, Job & Recruitment, Personal, Support Tickets, Other

Rules:
1. If the sender is a newsletter, use 'Newsletters'.
2. If it's a notification from LinkedIn/GitHub/Twitter, use 'Social Media' or 'Internal Updates'.
3. AVOID 'Other' unless absolutely necessary.
4. Return ONLY a JSON array of objects.

Senders:
{json.dumps(sender_list, indent=2)}
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip any markdown code fences
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()
        
        # Sometimes Gemini adds text before or after JSON
        json_match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
            
        categorised = json.loads(raw)
        for item in categorised:
            em_key = str(item.get("email", "")).lower()
            cat_val = item.get("category", "Other")
            if em_key in groups:
                groups[em_key]["category"] = cat_val
        state["categorization_error"] = None
    except Exception as exc:
        state["categorization_error"] = f"Categorization failed: {exc}"
        # Fallback logic: if it's newsletter-like or common domains
        for se, info in groups.items():
            if "news" in se or "info@" in se:
                info["category"] = "Newsletters"
            elif "promo" in se or "deal" in se:
                info["category"] = "Promotions"

    state["sender_groups"] = groups
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 – APPLY VIP TAGS
# ─────────────────────────────────────────────────────────────────────────────
def apply_vip_tags_node(state: InboxState) -> InboxState:
    """Mark senders that match VIP keywords via substring search on email + name."""
    vip_keywords = [v.strip().lower() for v in state.get("vip_senders", []) if v.strip()]
    groups = state.get("sender_groups", {})
    for se, info in groups.items():
        se_lower = se.lower()
        name_lower = info.get("name", "").lower()
        for kw in vip_keywords:
            # Match if keyword appears anywhere in email address OR display name
            if kw in se_lower or kw in name_lower:
                info["is_vip"] = True
                break
    state["sender_groups"] = groups
    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 – EXECUTE ACTION (Move / Delete)
# ─────────────────────────────────────────────────────────────────────────────
def execute_action_node(state: InboxState) -> InboxState:
    """Perform the IMAP move or delete for a group of UIDs."""
    action = state.get("action")
    category_key = state.get("action_category", "")
    folder_name = state.get("action_folder", "")
    groups = state.get("sender_groups", {})

    # Collect UIDs for the target category (or VIP if key == "__VIP__")
    uids = []
    if category_key == "__VIP__":
        for info in groups.values():
            if info.get("is_vip"):
                uids.extend(info.get("uids", []))
    else:
        for info in groups.values():
            if info.get("category") == category_key and not info.get("is_vip"):
                uids.extend(info.get("uids", []))

    if not uids:
        state["error"] = "No UIDs found for this group."
        return state

    try:
        mail = _imap_connect()
        mail.select("INBOX")
        uid_str = ",".join(uids)

        if action == "move":
            # Create folder if it doesn't exist
            status, _ = mail.create(folder_name)
            # (Gmail may return NO if it already exists — that's fine)
            # Copy emails to destination
            mail.uid("copy", uid_str, folder_name)
            # Delete from INBOX
            mail.uid("store", uid_str, "+FLAGS", r"(\Deleted)")
            mail.expunge()
            state["error"] = None

        elif action == "delete":
            # Move to [Gmail]/Trash
            trash_folder = "[Gmail]/Trash"
            mail.uid("copy", uid_str, trash_folder)
            mail.uid("store", uid_str, "+FLAGS", r"(\Deleted)")
            mail.expunge()
            state["error"] = None

        mail.logout()

        # Remove acted-upon senders from groups so they disappear from UI
        acted_emails = []
        if category_key == "__VIP__":
            acted_emails = [e for e, i in groups.items() if i.get("is_vip")]
        else:
            acted_emails = [e for e, i in groups.items()
                            if i.get("category") == category_key and not i.get("is_vip")]
        for em in acted_emails:
            del groups[em]
        state["sender_groups"] = groups

    except Exception as exc:
        state["error"] = f"Action error: {exc}"

    return state


# ─────────────────────────────────────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(InboxState)
    builder.add_node("fetch_emails", fetch_emails_node)
    builder.add_node("categorise_senders", categorise_senders_node)
    builder.add_node("apply_vip_tags", apply_vip_tags_node)
    builder.add_node("execute_action", execute_action_node)

    builder.set_entry_point("fetch_emails")
    builder.add_edge("fetch_emails", "categorise_senders")
    builder.add_edge("categorise_senders", "apply_vip_tags")
    builder.add_edge("apply_vip_tags", END)
    builder.add_edge("execute_action", END)

    return builder.compile()


GRAPH = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "sender_groups" not in st.session_state:
    st.session_state.sender_groups = {}
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False
if "error" not in st.session_state:
    st.session_state.error = None
if "action_result" not in st.session_state:
    st.session_state.action_result = None
if "selected_senders" not in st.session_state:
    st.session_state.selected_senders = {}
if "global_rag" not in st.session_state:
    st.session_state.global_rag = None
if "category_rags" not in st.session_state:
    st.session_state.category_rags = {}
if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = []
if "last_cost_info" not in st.session_state:
    st.session_state.last_cost_info = None


# ── RAG HELPERS ──────────────────────────────────────────────────────────────
def _get_email_body(mail: imaplib.IMAP4_SSL, uid: str) -> str:
    """Fetch and clean full email body."""
    status, data = mail.uid("fetch", uid, "(RFC822)")
    if status != "OK":
        return ""
    raw_email = data[0][1]
    msg = email.message_from_bytes(raw_email)
    
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                body += part.get_payload(decode=True).decode(errors="replace")
            elif content_type == "text/html" and "attachment" not in content_disposition:
                html = part.get_payload(decode=True).decode(errors="replace")
                soup = BeautifulSoup(html, "lxml")
                body += soup.get_text(separator=" ")
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            body = msg.get_payload(decode=True).decode(errors="replace")
        elif content_type == "text/html":
            html = msg.get_payload(decode=True).decode(errors="replace")
            soup = BeautifulSoup(html, "lxml")
            body = soup.get_text(separator=" ")
            
    return body[:2000]  # Truncate as requested


def calculate_and_display_cost(all_email_text: str, label="last_cost_info"):
    """Calculate and store estimated Gemini API cost (Gemini 2.0 Flash rates)."""
    try:
        # 1. Initialize the Gemini model just for counting
        model = genai.GenerativeModel("gemini-2.0-flash") 
        
        # 2. Count the exact number of tokens
        token_count = model.count_tokens(all_email_text).total_tokens
        
        # 3. Calculate the INR cost
        cost_per_million_inr = 8.30
        total_cost_inr = (token_count / 1_000_000) * cost_per_million_inr
        
        # 4. Store in session state for persistence
        st.session_state[label] = {
            "tokens": token_count,
            "cost": total_cost_inr,
            "actual": True
        }
        
        return total_cost_inr
    except Exception as e:
        # Fallback persistence so the UI knows we tried
        if label not in st.session_state:
            st.session_state[label] = {
                "tokens": 0,
                "cost": 0.0,
                "actual": False,
                "error": str(e)
            }
        return 0.0

def estimate_cost_from_metadata(uids):
    """Estimate cost based on RFC822.SIZE without fetching full bodies."""
    if not st.session_state.get("raw_emails"):
        return 0.0, 0
    
    # Map UID to size
    uid_to_size = {str(e['uid']): e.get('size_bytes', 0) for e in st.session_state.raw_emails}
    total_size = sum(uid_to_size.get(str(u), 0) for u in uids)
    
    # Heuristic: 1 token per 5 bytes (considers headers and overhead)
    est_tokens = int(total_size / 5)
    cost_per_million_inr = 8.30
    est_cost = (est_tokens / 1_000_000) * cost_per_million_inr
    
    return est_cost, est_tokens


def _initialize_rag(uids: List[str], label: str, limit: Optional[int] = None, cost_label: str = "last_cost_info"):
    """Load emails, embed, and store in FAISS. If limit is provided, only processes that many."""
    if not uids:
        return None
    
    if limit:
        uids = uids[:limit]
    
    total_count = len(uids)
    docs = []
    status_text = f"Loading RAG for {label} ({total_count} emails)..."
    progress_bar = st.progress(0, text=status_text)
    
    try:
        mail = _imap_connect()
        mail.select("INBOX")
        
        for idx, uid in enumerate(uids):
            body = _get_email_body(mail, uid)
            if body.strip():
                docs.append(Document(page_content=body, metadata={"uid": uid}))
            
            # Update progress every 5 emails or at the end
            if idx % 5 == 0 or idx == total_count - 1:
                progress = (idx + 1) / total_count
                progress_bar.progress(progress, text=f"Processed {idx+1}/{total_count} emails...")
        
        mail.logout()
        
        if not docs:
            st.warning("No readable content found in these emails.")
            return None
            
        # --- COST ESTIMATION STEP ---
        all_text = "\n".join([d.page_content for d in docs])
        calculate_and_display_cost(all_text, label=cost_label)
        # ----------------------------
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)
        
        # Batching logic to avoid 429 quota errors
        batch_size = 20  # Reduced for more stable compliance
        vectorstore = None
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            progress = (i + len(batch)) / total_count
            progress_bar.progress(progress, text=f"Embedding batch {i//batch_size + 1}... ({i+len(batch)}/{total_count})")
            
            # Proactive sleep to avoid hitting RPM limits
            if i > 0:
                time.sleep(2)  # Mandatory wait between batches
            
            # Simple retry loop with exponential backoff
            retry_count = 0
            max_retries = 10  # Increased for extra resilience
            while retry_count < max_retries:
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        batch_vs = FAISS.from_documents(batch, embeddings)
                        vectorstore.merge_from(batch_vs)
                    break # Success, move to next batch
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        retry_count += 1
                        wait_time = (2 ** retry_count) + 5  # Longer base wait
                        st.warning(f"⚠️ Rate limit hit. Waiting {wait_time}s to retry (Attempt {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        raise e # Re-raise non-quota errors
            else:
                raise Exception(f"Failed to embed batch after {max_retries} retries due to quota limits.")
            
        progress_bar.empty()
        st.success(f"RAG Engine loaded for {label}!")
        return vectorstore
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"RAG Error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scan Settings")
    st.markdown("---")

    today = date.today()
    five_years_ago = today - timedelta(days=5 * 365)

    start_date = st.date_input(
        "📅 From Date",
        value=today - timedelta(days=90),
        min_value=five_years_ago,
        max_value=today,
        key="start_date",
    )
    end_date = st.date_input(
        "📅 To Date",
        value=today,
        min_value=five_years_ago,
        max_value=today,
        key="end_date",
    )

    st.markdown("---")
    vip_input = st.text_input(
        "⭐ VIP Senders",
        placeholder="e.g. boss@company.com, cto@startup.io",
        help="Comma-separated email addresses to highlight as VIP",
        key="vip_input",
    )

    st.markdown("---")
    unread_only = st.checkbox(
        "🔓 Only show unread emails",
        value=True,
        help="Uncheck to find all emails in the date range. Note: Scanning all emails may take longer.",
        key="unread_only",
    )

    st.markdown("---")
    scan_clicked = st.button("🔍 Scan Inbox", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        f"<small style='color:rgba(200,200,220,0.45);'>Scanning: **{GMAIL_ADDRESS}**</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📬 AI Inbox Declutter Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Powered by Gemini · LangGraph · FAISS RAG · Gmail — scan, categorise, and chat with your inbox.</div>',
    unsafe_allow_html=True,
)

# ── GLOBAL RAG UI ─────────────────────────────────────────────────────────────
if st.session_state.scan_done and st.session_state.sender_groups:
    all_uids = [em["uid"] for em in st.session_state.raw_emails]
    
    # Global cost display: Actual if loaded, Estimate if not
    if st.session_state.get("last_cost_info"):
        ci = st.session_state.last_cost_info
        st.info(f"📊 **Global RAG Loaded:** {ci['tokens']:,} tokens | 💰 **Actual Cost:** ₹{ci['cost']:.4f}")
    else:
        est_cost_glob, est_tokens_glob = estimate_cost_from_metadata(all_uids)
        st.caption(f"📋 **Total Inbox Estimate to load:** ~{est_tokens_glob:,} tokens | ~₹{est_cost_glob:.4f}")

    total_inbox_size_bytes = sum(em.get("size_bytes", 0) for em in st.session_state.raw_emails)
    total_inbox_mb = total_inbox_size_bytes / (1024 * 1024)
    
    col_rag1, col_rag2 = st.columns([1, 1])
    with col_rag1:
        st.markdown(f"📦 **Total Data Found:** {total_inbox_mb:.1f} MB")
        if st.button(f"🚀 Load Global Inbox RAG ({len(all_uids)} emails)", use_container_width=True, type="primary"):
            st.session_state.global_rag = _initialize_rag(all_uids, "Global Inbox")
            st.rerun() # Ensure cost display updates
    
    if st.session_state.global_rag:
        with st.expander("💬 Chat with Global Inbox", expanded=True):
            user_q = st.text_input("Ask anything about your inbox...", key="global_rag_q")
            if user_q:
                with st.spinner("Thinking..."):
                    docs = st.session_state.global_rag.similarity_search(user_q, k=4)
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"Context from emails:\n{context}\n\nQuestion: {user_q}\n\nAnswer concisely based on the context."
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
    else:
        st.info("💡 **Tip:** Click 'Load Global Inbox RAG' above to enable chat for your entire inbox. This database will persist even if you adjust settings (until you restart the app).")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCAN HANDLER
# ─────────────────────────────────────────────────────────────────────────────
if scan_clicked:
    if start_date > end_date:
        st.error("⚠️ Start date must be before End date.")
    else:
        vip_list = [v.strip() for v in vip_input.split(",") if v.strip()]
        initial_state: InboxState = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "vip_senders": vip_list,
            "raw_emails": [],
            "sender_groups": {},
            "error": None,
            "action": None,
            "action_category": None,
            "action_folder": None,
            "has_attachments": False,
            "attachment_size_mb": 0.0,
            "total_uids_found": 0,
            "is_fallback_search": False,
            "categorization_error": None,
            "unread_only": unread_only,
        }
        with st.status("🚀 Processing your inbox...", expanded=True) as status:
            status.update(label="🔄 Connecting to Gmail and fetching emails...", state="running")
            # Step 1: Fetch
            fetch_res = fetch_emails_node(initial_state)
            
            if fetch_res.get("error"):
                status.update(label="❌ Fetch Error", state="error")
                result = fetch_res
            elif not fetch_res.get("raw_emails"):
                status.update(label="✅ No emails found.", state="complete")
                result = fetch_res
            else:
                status.update(label=f"📊 Found {len(fetch_res['raw_emails'])} emails. Categorising with AI...", state="running")
                # Step 2: Categorise
                cat_res = categorise_senders_node(fetch_res)
                
                status.update(label="⭐ Applying VIP filters...", state="running")
                # Step 3: VIP Tags
                final_res = apply_vip_tags_node(cat_res)
                
                status.update(label="✅ Scan Complete!", state="complete")
                result = final_res

        st.session_state.sender_groups = result.get("sender_groups", {})
        st.session_state.raw_emails = result.get("raw_emails", [])
        st.session_state.error = result.get("error")
        st.session_state.total_uids_found = result.get("total_uids_found", 0)
        st.session_state.is_fallback_search = result.get("is_fallback_search", False)
        st.session_state.categorization_error = result.get("categorization_error")
        st.session_state.scan_done = True
        st.session_state.action_result = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"❌ {st.session_state.error}")

if st.session_state.action_result:
    if st.session_state.action_result.get("ok"):
        st.toast(f"✅ {st.session_state.action_result['msg']}", icon="✅")
    else:
        st.toast(f"❌ {st.session_state.action_result['msg']}", icon="❌")
    st.session_state.action_result = None


def _do_action(action: str, category_key: str, folder: str = ""):
    """Bulk action for a whole category (or VIP group)."""
    groups = st.session_state.sender_groups
    vip_list = [v.strip() for v in st.session_state.get("vip_input", "").split(",") if v.strip()]

    action_state: InboxState = {
        "start_date": st.session_state.start_date.isoformat() if hasattr(st.session_state.start_date, "isoformat") else str(st.session_state.start_date),
        "end_date": st.session_state.end_date.isoformat() if hasattr(st.session_state.end_date, "isoformat") else str(st.session_state.end_date),
        "vip_senders": vip_list,
        "raw_emails": [],
        "sender_groups": groups,
        "error": None,
        "action": action,
        "action_category": category_key,
        "action_folder": folder,
        "has_attachments": False,
        "attachment_size_mb": 0.0,
        "total_uids_found": 0,
        "is_fallback_search": False,
        "categorization_error": None,
    }
    
    verb_ing = "Moving" if action == "move" else "Deleting"
    label = "VIP group" if category_key == "__VIP__" else category_key
    st.toast(f"⏳ {verb_ing} emails in '{label}'. Please wait...", icon="⏳")
    time.sleep(0.1)
    with st.spinner(f"⏳ {verb_ing} emails in '{label}'... This may take a moment."):
        result = execute_action_node(action_state)
        st.session_state.sender_groups = result.get("sender_groups", {})
        err = result.get("error")
    if err:
        st.session_state.action_result = {"ok": False, "msg": err}
    else:
        verb = "moved" if action == "move" else "deleted"
        label = "VIP group" if category_key == "__VIP__" else category_key
        msg = f"All emails in '{label}' successfully {verb}."
        if action == "move":
            msg = f"All emails in '{label}' moved to folder '{folder}'."
        st.session_state.action_result = {"ok": True, "msg": msg}
    st.rerun()


def _do_sender_action(sender_email: str, action: str, folder: str = ""):
    """Perform move or delete for a SINGLE sender's emails."""
    groups = st.session_state.sender_groups
    sender_info = groups.get(sender_email)
    if not sender_info:
        st.session_state.action_result = {"ok": False, "msg": f"Sender {sender_email} not found."}
        st.rerun()
        return

    uids = sender_info.get("uids", [])
    if not uids:
        st.session_state.action_result = {"ok": False, "msg": "No emails found for this sender."}
        st.rerun()
        return

    uid_str = ",".join(uids)
    
    verb_ing = "Moving" if action == "move" else "Deleting"
    st.toast(f"⏳ {verb_ing} emails from '{sender_info['name']}'. Please wait...", icon="⏳")
    time.sleep(0.1)
    with st.spinner(f"⏳ {verb_ing} {sender_info['count']} email(s) from '{sender_info['name']}'..."):
        try:
            mail = _imap_connect()
            mail.select("INBOX")
            if action == "move":
                mail.create(folder)   # OK if already exists
                mail.uid("copy", uid_str, folder)
                mail.uid("store", uid_str, "+FLAGS", r"(\Deleted)")
                mail.expunge()
                msg = f"{sender_info['count']} email(s) from '{sender_info['name']}' moved to '{folder}'."
            else:
                mail.uid("copy", uid_str, "[Gmail]/Trash")
                mail.uid("store", uid_str, "+FLAGS", r"(\Deleted)")
                mail.expunge()
                msg = f"{sender_info['count']} email(s) from '{sender_info['name']}' moved to Trash."
            mail.logout()
            # Remove sender from groups
            del groups[sender_email]
            st.session_state.sender_groups = groups
            st.session_state.action_result = {"ok": True, "msg": msg}
        except Exception as exc:
            st.session_state.action_result = {"ok": False, "msg": f"Error: {exc}"}
    st.rerun()

def _do_multi_sender_action(action: str, sender_emails: list, folder: str = None):
    """Processes multiple selected senders in a batch."""
    if not sender_emails:
        st.warning("No senders selected.")
        return

    groups = st.session_state.sender_groups
    all_uids = []
    sender_names = []
    
    for se in sender_emails:
        if se in groups:
            info = groups[se]
            all_uids.extend(info.get("uids", []))
            sender_names.append(info.get("name", se))

    if not all_uids:
        st.error("No emails found for selected senders.")
        return

    verb_ing = "Moving" if action == "move" else "Deleting"
    count = len(all_uids)
    names_summary = ", ".join(sender_names[:3]) + ("..." if len(sender_names) > 3 else "")
    
    st.toast(f"⏳ {verb_ing} {count} emails from {len(sender_emails)} senders...", icon="⏳")
    time.sleep(0.1)
    
    with st.spinner(f"⏳ {verb_ing} {count} emails from {len(sender_emails)} senders: {names_summary}"):
        try:
            mail = _imap_connect()
            mail.select("INBOX")
            uid_str = ",".join(all_uids)
            
            if action == "delete":
                # Copy to Trash first if possible
                try:
                    mail.uid("COPY", uid_str, "[Gmail]/Trash")
                    mail.uid("STORE", uid_str, "+FLAGS", r"(\Deleted)")
                except:
                    # Fallback to just marking deleted
                    mail.uid("STORE", uid_str, "+FLAGS", r"(\Deleted)")
                mail.expunge()
                msg = f"Successfully deleted {count} emails from {len(sender_emails)} senders."
            elif action == "move" and folder:
                mail.create(folder)
                mail.uid("COPY", uid_str, folder)
                mail.uid("STORE", uid_str, "+FLAGS", r"(\Deleted)")
                mail.expunge()
                msg = f"Successfully moved {count} emails from {len(sender_emails)} senders to '{folder}'."
            
            mail.logout()
            
            # Update state
            new_groups = groups.copy()
            for se in sender_emails:
                if se in new_groups:
                    del new_groups[se]
                if se in st.session_state.selected_senders:
                    st.session_state.selected_senders[se] = False
            
            st.session_state.sender_groups = new_groups
            st.session_state.action_result = {"ok": True, "msg": msg}
            st.success(f"✅ {msg}")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during bulk action: {e}")
            st.session_state.action_result = {"ok": False, "msg": f"Error: {e}"}


if st.session_state.scan_done and st.session_state.sender_groups:
    groups = st.session_state.sender_groups

    total_senders = len(groups)
    total_emails = sum(g["count"] for g in groups.values())
    vip_count = sum(1 for g in groups.values() if g.get("is_vip"))
    categories = sorted({g["category"] for g in groups.values() if not g.get("is_vip")})

    # ── Metrics ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{total_emails}</div><div class="metric-label">Processed (of {st.session_state.get("total_uids_found", 0)})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{total_senders}</div><div class="metric-label">Unique Senders</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(categories)}</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
    with c4:
        total_mb = sum(em.get("size_bytes", 0) for em in st.session_state.raw_emails) / (1024 * 1024)
        st.markdown(f'<div class="metric-box"><div class="metric-val">{total_mb:.1f} MB</div><div class="metric-label">Total Size</div></div>', unsafe_allow_html=True)

    if st.session_state.get("is_fallback_search"):
        st.info("⚠️ **Note:** No unread emails were found in your selected date range, so we are showing **all current unread emails** from your Inbox.")

    if st.session_state.get("total_uids_found", 0) > len(st.session_state.raw_emails):
        st.info(f"ℹ️ **Note:** IMAP search found {st.session_state['total_uids_found']} unread emails, but only {len(st.session_state.raw_emails)} were successfully fetched/processed in this range. This can happen if some emails have empty headers or invalid UIDs.")

    if st.session_state.get("categorization_error"):
        st.warning(f"🤖 **Categorization Info:** {st.session_state['categorization_error']}. Some emails may be listed as 'Uncategorised' or fallback categories.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── VIP SECTION ───────────────────────────────────────────────────────────
    vip_senders = {e: g for e, g in groups.items() if g.get("is_vip")}
    if vip_senders:
        st.markdown("## ⭐ VIP Senders")
        total_vip_emails = sum(g["count"] for g in vip_senders.values())
        for se, info in vip_senders.items():
            st.markdown(
                f"""<div class="vip-card">
                <span class="vip-badge">VIP</span>
                <strong>{info['name']}</strong>
                &nbsp;<span style="color:rgba(200,200,220,0.55);font-size:0.82rem;">&lt;{se}&gt;</span>
                &nbsp;·&nbsp;<span style="color:#fbbf24;font-size:0.85rem;">{info['category']}</span>
                &nbsp;·&nbsp;<span style="color:rgba(200,200,220,0.7);font-size:0.82rem;">{info['count']} email{'s' if info['count']>1 else ''}</span>
                &nbsp;·&nbsp;<span style="color:#fbbf24;font-size:0.82rem;">📎 {sum(em.get('size_bytes', 0) for em in st.session_state.raw_emails if em['sender_email'] == se)/(1024*1024):.1f} MB</span>
                <br><small style="color:rgba(200,200,220,0.45);">{'  ·  '.join(info['subjects'][:3])}</small>
                </div>""",
                unsafe_allow_html=True,
            )

        # VIP action panel — always visible
        st.markdown('<div class="vip-action-panel">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="action-panel-label">⭐ Actions for all {total_vip_emails} VIP emails</div>',
            unsafe_allow_html=True,
        )
        vc1, vc2, vc3 = st.columns([3, 1, 1])
        with vc1:
            st.text_input(
                "Create new folder or type existing folder name",
                key="vip_folder",
                placeholder="e.g. VIP / Important",
            )
        with vc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📁 Move to Folder", key="vip_move", use_container_width=True):
                fv = st.session_state.get("vip_folder", "").strip()
                if fv:
                    _do_action("move", "__VIP__", fv)
                else:
                    st.warning("Please type a folder name first.")
        with vc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if not st.session_state.get("confirm_del___VIP__"):
                if st.button("🗑️ Delete All", key="vip_del_btn", use_container_width=True):
                    st.session_state["confirm_del___VIP__"] = True
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="del-confirm">⚠️ Move {total_vip_emails} VIP emails to Trash?</div>',
                    unsafe_allow_html=True,
                )
                cy, cn = st.columns(2)
                with cy:
                    if st.button("✅ Confirm", key="vip_del_yes", use_container_width=True, type="primary"):
                        st.session_state["confirm_del___VIP__"] = False
                        _do_action("delete", "__VIP__")
                with cn:
                    if st.button("❌ Cancel", key="vip_del_no", use_container_width=True):
                        st.session_state["confirm_del___VIP__"] = False
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CATEGORY SECTIONS ─────────────────────────────────────────────────────
    st.markdown("## 📂 Email Categories")
    st.caption("▸ Expand a category to preview senders and act on individual senders · The action bar below each category handles bulk actions")
    
    col_sort, col_expand = st.columns([3, 1])
    with col_sort:
        sort_order = st.selectbox(
            "Sort categories by Email Count",
            options=["Descending", "Ascending"],
            index=0,
            key="cat_sort_order"
        )
    with col_expand:
        st.markdown("<br>", unsafe_allow_html=True)
        # Default to True as requested by user to keep things open
        expand_all = st.toggle("Expand all", value=True, key="expand_all")
    st.markdown("")

    CAT_EMOJI = {
        "Newsletters": "📰", "Promotions": "🎁", "Social Media": "💬",
        "Internal Updates": "🏢", "Finance & Billing": "💳",
        "Travel & Bookings": "✈️", "Spam": "🚫", "Shopping": "🛍️",
        "Job & Recruitment": "💼", "Personal": "👤",
        "Support Tickets": "🎟️", "Other": "📌", "Uncategorised": "❓",
    }

    # Calculate email counts mapped to each category to establish sorting order
    category_counts = []
    for cat in categories:
        cat_senders = {e: g for e, g in groups.items() if g.get("category") == cat and not g.get("is_vip")}
        cat_email_count = sum(g["count"] for g in cat_senders.values())
        
        # Get raw emails matching these senders to calculate total attachment info
        cat_emails = [em for em in st.session_state.get("raw_emails", []) if em["sender_email"] in cat_senders]
        cat_has_att = any(em.get("has_attachment") for em in cat_emails)
        cat_att_size = sum(em.get("size_bytes", 0) for em in cat_emails) / (1024 * 1024)
        
        if cat_senders:
            category_counts.append({
                "category": cat, 
                "count": cat_email_count, 
                "senders": cat_senders,
                "has_att": cat_has_att,
                "att_size": cat_att_size
            })

    # Sort based on user selection
    reverse_sort = sort_order == "Descending"
    sorted_categories = sorted(category_counts, key=lambda x: x["count"], reverse=reverse_sort)

    for cat_data in sorted_categories:
        cat = cat_data["category"]
        cat_email_count = cat_data["count"]
        cat_senders = cat_data["senders"]
        cat_sender_count = len(cat_senders)
        emoji = CAT_EMOJI.get(cat, "📌")
        safe_key = cat.replace(" ", "_").replace("&", "and")
        confirm_key = f"confirm_del_{safe_key}"
        # ── Sender list inside COLLAPSIBLE expander (+ per-sender quick actions)
        att_info = f" · 📎 Attachments: Yes ({cat_data['att_size']:.1f} MB)" if cat_data["has_att"] else " · 📎 Attachments: No"
        with st.expander(
            f"{emoji} **{cat}** — {cat_sender_count} sender{'s' if cat_sender_count>1 else ''} · {cat_email_count} email{'s' if cat_email_count>1 else ''}{att_info}",
            expanded=expand_all,
        ):
            # Category RAG UI
            cat_uids = []
            for s in cat_senders.values():
                cat_uids.extend(s.get("uids", []))
            
            # Category-specific cost estimation
            est_cost_cat, est_tokens_cat = estimate_cost_from_metadata(cat_uids)
            
            # If already loaded, show actual cost, otherwise show estimate
            cat_rag_key = f"cost_info_{safe_key}"
            if cat_rag_key in st.session_state:
                ci_cat = st.session_state[cat_rag_key]
                st.info(f"📊 **Loaded Data:** {ci_cat['tokens']:,} tokens | 💰 **Actual Cost:** ₹{ci_cat['cost']:.4f}")
            else:
                st.caption(f"📋 **Estimated to load:** ~{est_tokens_cat:,} tokens | ~₹{est_cost_cat:.4f}")

            if st.button(f"🧠 Load RAG for {cat}", key=f"rag_btn_{safe_key}", use_container_width=True):
                st.session_state.category_rags[cat] = _initialize_rag(cat_uids, cat, cost_label=cat_rag_key)
                st.rerun() # Ensure cost display updates
            
            if cat in st.session_state.category_rags and st.session_state.category_rags[cat]:
                st.markdown(f"**Chat with {cat}**")
                # Using a unique key for the chat input per category
                cat_q = st.text_input("Ask about this category...", key=f"rag_q_{safe_key}")
                if cat_q:
                    with st.spinner("Analyzing emails..."):
                        docs = st.session_state.category_rags[cat].similarity_search(cat_q, k=3)
                        context = "\n\n".join([d.page_content for d in docs])
                        prompt = f"Context from '{cat}' emails:\n{context}\n\nQuestion: {cat_q}\n\nAnswer concisely based on context."
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        response = model.generate_content(prompt)
                        st.info(response.text)
            else:
                st.caption("💡 Click 'Load RAG' above to chat with emails in this category.")
            st.markdown("---")

            # Bulk Selection Helpers
            c_sel1, c_sel2, c_sel3 = st.columns([1, 1, 2])
            with c_sel1:
                if st.button("✅ Select All", key=f"sel_all_{safe_key}", use_container_width=True):
                    for se in cat_senders.keys():
                        se_safe = re.sub(r"[^a-zA-Z0-9]", "_", se)
                        st.session_state.selected_senders[se] = True
                        st.session_state[f"check_{safe_key}_{se_safe}"] = True
                    st.rerun()
            with c_sel2:
                if st.button("❌ Select None", key=f"sel_none_{safe_key}", use_container_width=True):
                    for se in cat_senders.keys():
                        se_safe = re.sub(r"[^a-zA-Z0-9]", "_", se)
                        st.session_state.selected_senders[se] = False
                        st.session_state[f"check_{safe_key}_{se_safe}"] = False
                    st.rerun()
            st.markdown("---")

            sorted_senders = sorted(cat_senders.items(), key=lambda item: item[1]["count"], reverse=reverse_sort)
            for se, info in sorted_senders:
                se_safe = re.sub(r"[^a-zA-Z0-9]", "_", se)
                
                # Checkbox for multi-select
                is_selected = st.checkbox(
                    "Select", 
                    value=st.session_state.selected_senders.get(se, False), 
                    key=f"check_{safe_key}_{se_safe}",
                    label_visibility="collapsed"
                )
                st.session_state.selected_senders[se] = is_selected

                # Sender header (indented slightly if we have a checkbox)
                st.markdown(
                    f"**{info['name']}** "
                    f"<span style='color:rgba(200,200,220,0.5);font-size:0.8rem;'>&lt;{se}&gt;</span> — "
                    f"<span style='color:#60a5fa;'>{info['count']} email{'s' if info['count']>1 else ''}</span>",
                    unsafe_allow_html=True,
                )
                if info["subjects"]:
                    subjects_md = " &nbsp;|&nbsp; ".join(
                        f"<span style='font-size:0.78rem;color:rgba(200,200,220,0.55);'>{s}</span>"
                        for s in info["subjects"]
                    )
                    st.markdown(f"⤷ {subjects_md}", unsafe_allow_html=True)

                # Per-sender mini action row (inline, quick)
                st.markdown('<div class="sender-mini-action">', unsafe_allow_html=True)
                st.markdown('<div class="sender-mini-label">Quick action — this sender only</div>', unsafe_allow_html=True)
                sa1, sa2, sa3 = st.columns([3, 1, 1])
                with sa1:
                    st.text_input(
                        "Folder",
                        key=f"sfolder_{safe_key}_{se_safe}",
                        placeholder=f"e.g. Archive/{cat}",
                        label_visibility="collapsed",
                    )
                with sa2:
                    if st.button("📁 Move", key=f"smove_{safe_key}_{se_safe}", use_container_width=True):
                        sv = st.session_state.get(f"sfolder_{safe_key}_{se_safe}", "").strip()
                        if sv:
                            _do_sender_action(se, "move", sv)
                        else:
                            st.warning("Type a folder name first.")
                with sa3:
                    sdel_key = f"sdel_confirm_{safe_key}_{se_safe}"
                    if not st.session_state.get(sdel_key):
                        if st.button("🗑️ Delete", key=f"sdel_{safe_key}_{se_safe}", use_container_width=True):
                            st.session_state[sdel_key] = True
                            st.rerun()
                    else:
                        if st.button("✅ Confirm", key=f"sdel_yes_{safe_key}_{se_safe}", use_container_width=True, type="primary"):
                            st.session_state[sdel_key] = False
                            _do_sender_action(se, "delete")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

        # ── Bulk action panel — ALWAYS VISIBLE beneath the expander ──────────
        # Check if any senders are selected for this category
        selected_in_cat = [se for se in cat_senders.keys() if st.session_state.selected_senders.get(se)]
        num_selected = len(selected_in_cat)

        st.markdown('<div class="action-panel">', unsafe_allow_html=True)
        if num_selected > 0:
            st.markdown(
                f'<div class="action-panel-label" style="background:rgba(96,165,250,0.1); border-left:4px solid #60a5fa;">🎯 {num_selected} sender(s) selected — act on {num_selected} groups only</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="action-panel-label">{emoji} {cat} — bulk action on all {cat_email_count} emails</div>',
                unsafe_allow_html=True,
            )

        ap1, ap2, ap3 = st.columns([3, 1, 1])
        with ap1:
            st.text_input(
                "Create new folder or type existing folder name",
                key=f"folder_{safe_key}",
                placeholder=f"e.g. Archive/{cat}",
                label_visibility="visible",
            )
        with ap2:
            st.markdown("<br>", unsafe_allow_html=True)
            move_label = "📁 Move Selected" if num_selected > 0 else "📁 Move All to Folder"
            if st.button(move_label, key=f"move_{safe_key}", use_container_width=True):
                folder_val = st.session_state.get(f"folder_{safe_key}", "").strip()
                if folder_val:
                    if num_selected > 0:
                        _do_multi_sender_action("move", selected_in_cat, folder_val)
                    else:
                        _do_action("move", cat, folder_val)
                else:
                    st.warning("⚠️ Please type a folder name above first.")
        with ap3:
            st.markdown("<br>", unsafe_allow_html=True)
            del_label = "🗑️ Delete Selected" if num_selected > 0 else "🗑️ Delete All"
            
            if not st.session_state.get(confirm_key):
                if st.button(del_label, key=f"del_{safe_key}", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                confirm_msg = f"⚠️ Move {num_selected} selected sender groups to Trash?" if num_selected > 0 else f"⚠️ Move all {cat_email_count} emails to Trash?"
                st.markdown(
                    f'<div class="del-confirm">{confirm_msg}</div>',
                    unsafe_allow_html=True,
                )
                cy, cn = st.columns(2)
                with cy:
                    if st.button("✅ Yes, Trash", key=f"del_yes_{safe_key}", use_container_width=True, type="primary"):
                        st.session_state[confirm_key] = False
                        if num_selected > 0:
                            _do_multi_sender_action("delete", selected_in_cat)
                        else:
                            _do_action("delete", cat)
                with cn:
                    if st.button("❌ Cancel", key=f"del_no_{safe_key}", use_container_width=True):
                        st.session_state[confirm_key] = False
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.scan_done and not st.session_state.sender_groups:
    st.success("🎉 Great news — no unread emails found in that date range!")
else:
    # Landing state
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 2rem;opacity:0.55;">
            <div style="font-size:4rem;">📬</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;">Configure your scan settings in the sidebar and click <strong>Scan Inbox</strong> to get started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
