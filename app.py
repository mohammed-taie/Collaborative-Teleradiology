import os
import io
import zipfile
import shutil
import sqlite3
import warnings
import pathlib
import json
from datetime import datetime
import numpy as np
import streamlit as st
import pydicom
from PIL import Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# --- CONFIGURATION & SECRETS
# ----------------------------
st.set_page_config(page_title="🏥 AI DICOM Review", layout="wide")

# File paths (can still be overridden via environment variables if needed)
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH  = os.getenv("DB_PATH",  "cases.db")

# Hardcoded simple credentials
ADMIN_USER = "admin"
ADMIN_PASS = "password"

REPORT_TEMPLATES   = ["Free-Text", "BI-RADS", "PI-RADS"]
CANVAS_MAX_WIDTH   = 512

# ensure storage
os.makedirs(DATA_DIR, exist_ok=True)

# suppress benign pydicom warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydicom.valuerep",
    message="Invalid value for VR UI.*"
)

# ----------------------------
# --- DATABASE INITIALIZATION
# ----------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code      TEXT,
            study_uid         TEXT UNIQUE,
            modality          TEXT,
            upload_date       TEXT,
            status            TEXT,
            assigned_to       TEXT,
            original_filename TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS comments (
            id        INTEGER PRIMARY KEY,
            case_id   INTEGER,
            user      TEXT,
            text      TEXT,
            timestamp TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS reports (
            id          INTEGER PRIMARY KEY,
            case_id     INTEGER,
            user        TEXT,
            report_text TEXT,
            timestamp   TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS collaborations (
            id           INTEGER PRIMARY KEY,
            case_id      INTEGER,
            user         TEXT,
            collaborator TEXT,
            timestamp    TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        );
        CREATE TABLE IF NOT EXISTS annotations (
            id         INTEGER PRIMARY KEY,
            case_id    INTEGER,
            user       TEXT,
            shape_type TEXT,
            shape_data TEXT,
            timestamp  TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        );
        """)
        conn.commit()

init_db()

# ----------------------------
# --- UTILITY FUNCTIONS
# ----------------------------
def safe_extract(zip_bytes: io.BytesIO, extract_to: pathlib.Path):
    """Extract zip file, preventing Zip-Slip."""
    with zipfile.ZipFile(zip_bytes) as z:
        for member in z.infolist():
            out_path = extract_to / pathlib.Path(member.filename).name
            if not out_path.resolve().is_relative_to(extract_to.resolve()):
                raise RuntimeError(f"Unsafe path detected: {member.filename}")
        z.extractall(path=extract_to)

def anonymize_and_save(ds: pydicom.Dataset, out_path: pathlib.Path):
    """Strip PHI and save DICOM safely."""
    ds.remove_private_tags()
    for tag in ["PatientName", "PatientID", "PatientBirthDate", "PatientAge",
                "InstitutionName", "ReferringPhysicianName", "StudyDescription"]:
        if hasattr(ds, tag):
            delattr(ds, tag)
    ds.save_as(str(out_path), write_like_original=False)

@st.cache_data
def load_dicom_pixel_array(path: str, mtime: float):
    ds = pydicom.dcmread(path)
    return ds.pixel_array

# ----------------------------
# --- AUTHENTICATION
# ----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.sidebar:
        st.header("🔒 Login")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if u == ADMIN_USER and p == ADMIN_PASS:
                st.session_state.authenticated = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

user = st.session_state.user
with st.sidebar:
    st.markdown(f"**Logged in as:** {user}")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Upload Study", "Reporting & Collaboration"])
    st.markdown("---")
    with st.expander("Annotation Mode"):
        annotate = st.checkbox("Enable Annotations")
        shape    = st.selectbox("Shape", ["freedraw", "rect", "circle", "polygon"])
        thickness= st.slider("Stroke width", 1, 10, 3)

# ----------------------------
# --- UPLOAD STUDY PAGE
# ----------------------------
if page == "Upload Study":
    st.header("📤 Upload DICOM Study (.zip)")
    zip_file   = st.file_uploader("ZIP of DICOM files", type="zip")
    assigned_to= st.text_input("Assign to Radiologist", value=user)
    if zip_file and st.button("Process Upload"):
        fname = zip_file.name
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            if conn.execute("SELECT 1 FROM cases WHERE original_filename=?", (fname,)).fetchone():
                st.info("This file has already been uploaded.")
                st.stop()
        try:
            with st.spinner("Extracting and anonymizing..."):
                with zipfile.ZipFile(zip_file) as zf:
                    sample      = pydicom.dcmread(io.BytesIO(zf.read(zf.infolist()[0].filename)))
                    study_uid   = sample.StudyInstanceUID
                    patient_code= sample.get("PatientID", "UNKNOWN")
                    modality    = sample.get("Modality", "OT")
                upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO cases
                          (patient_code, study_uid, modality, upload_date, status, assigned_to, original_filename)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (patient_code, study_uid, modality, upload_date, "new", assigned_to, fname))
                    case_id = cur.lastrowid
                    conn.commit()

                case_folder = pathlib.Path(DATA_DIR) / f"case_{case_id}"
                case_folder.mkdir(parents=True, exist_ok=True)
                safe_extract(zip_file, case_folder)

                for f in case_folder.glob("*"):
                    try:
                        ds = pydicom.dcmread(str(f))
                        anonymize_and_save(ds, f)
                    except Exception:
                        continue

                load_dicom_pixel_array.clear()
            st.success(f"Upload complete! Case ID: {case_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------------
# --- DASHBOARD PAGE
# ----------------------------
elif page == "Dashboard":
    st.title("📁 Case Dashboard")
    with st.sidebar.expander("Filters"):
        status_filter = st.selectbox("Status", ["All", "new", "in-review", "finalized"])
        my_only       = st.checkbox("My cases only", value=True)

    query   = "SELECT case_id, patient_code, modality, upload_date, status, assigned_to FROM cases"
    params  = []
    clauses = []
    if status_filter != "All":
        clauses.append("status = ?"); params.append(status_filter)
    if my_only:
        clauses.append("assigned_to = ?"); params.append(user)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY upload_date DESC"

    conn  = sqlite3.connect(DB_PATH)
    cases = conn.execute(query, params).fetchall()
    conn.close()

    if not cases:
        st.info("No cases found.")
    else:
        for cid, pc, md, ud, stt, asgn in cases:
            with st.expander(f"Case {cid} | {pc} | {md} | {stt} | {asgn}"):
                folder = pathlib.Path(DATA_DIR) / f"case_{cid}"
                dcms   = sorted(folder.glob("*.dcm"))
                if not dcms:
                    st.warning("No DICOM files.")
                    continue

                col1, col2 = st.columns([3, 1])
                with col1:
                    idx = st.slider("Slice", 0, len(dcms)-1, 0, key=f"s{cid}") if len(dcms)>1 else 0
                    arr = load_dicom_pixel_array(str(dcms[idx]), dcms[idx].stat().st_mtime)

                    if annotate:
                        scale = min(1, CANVAS_MAX_WIDTH/arr.shape[1])
                        w, h  = int(arr.shape[1]*scale), int(arr.shape[0]*scale)
                        # after
                        bg_pil = Image.fromarray(arr).resize((w, h))
                        canvas = st_canvas(
                            fill_color="rgba(0,0,0,0)",
                            stroke_width=thickness,
                            stroke_color="#FF0000",
                            background_image=bg_pil,      # ✅ PIL.Image
                            drawing_mode=shape,
                            key=f"c{cid}_{idx}",
                            width=w,
                            height=h
                        )
                        if st.button("Save Annotation", key=f"save{cid}_{idx}") and canvas.json_data:
                            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            inv = 1/scale
                            objs = []
                            for obj in canvas.json_data["objects"]:
                                t = obj.get("type")
                                if t in ("rect", "circle"):
                                    obj["left"]   *= inv
                                    obj["top"]    *= inv
                                    if t == "rect":
                                        obj["width"]  *= inv
                                        obj["height"] *= inv
                                    else:
                                        obj["radius"] *= inv
                                elif t == "polygon":
                                    for p in obj["points"]:
                                        p["x"] *= inv; p["y"] *= inv
                                objs.append(json.dumps(obj))
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.executemany(
                                    "INSERT INTO annotations(case_id,user,shape_type,shape_data,timestamp) VALUES (?,?,?,?,?)",
                                    [(cid, user, json.loads(o)["type"], o, ts) for o in objs]
                                )
                                conn.commit()
                            st.success("Annotation saved!")
                    else:
                        fig = px.imshow(arr, color_continuous_scale="gray")
                        fig.update_layout(dragmode="pan", margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

                with col2:
                    st.subheader("💬 Comments")
                    cm = st.text_input("Add comment:", key=f"cm{cid}")
                    if st.button("Post", key=f"p{cid}") and cm:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute(
                                "INSERT INTO comments(case_id,user,text,timestamp) VALUES (?,?,?,?)",
                                (cid, user, cm, ts)
                            )
                            conn.commit()
                        st.success("Comment added.")
                        st.rerun()
                    for usr, txt, ts in sqlite3.connect(DB_PATH).execute(
                        "SELECT user,text,timestamp FROM comments WHERE case_id=? ORDER BY timestamp", (cid,)
                    ).fetchall():
                        st.markdown(f"**{usr}** ({ts}): {txt}")

                    st.subheader("📄 Report")
                    last = sqlite3.connect(DB_PATH).execute(
                        "SELECT report_text FROM reports WHERE case_id=? ORDER BY timestamp DESC LIMIT 1", (cid,)
                    ).fetchone()
                    base = last[0] if last else ""
                    rpt  = st.text_area("Findings:", value=base, key=f"r{cid}")
                    if st.button("Save Report", key=f"sr{cid}"):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute(
                                "INSERT INTO reports(case_id,user,report_text,timestamp) VALUES (?,?,?,?)",
                                (cid, user, rpt, ts)
                            )
                            conn.execute("UPDATE cases SET status=? WHERE case_id=?", ("finalized", cid))
                            conn.commit()
                        st.success("Report saved & finalized.")
                        st.rerun()

# ----------------------------
# --- REPORTING & COLLABORATION
# ----------------------------
elif page == "Reporting & Collaboration":
    st.title("🤝 Reporting & Collaboration")
    cases = sqlite3.connect(DB_PATH).execute(
        "SELECT case_id,patient_code,modality,status FROM cases ORDER BY upload_date DESC"
    ).fetchall()
    if not cases:
        st.warning("No cases available.")
        st.stop()

    opts = {f"{c[0]} | {c[1]} | {c[2]} | {c[3]}": c[0] for c in cases}
    sel  = st.selectbox("Choose Case", list(opts.keys()))
    cid  = opts[sel]

    with st.sidebar.expander("Invite Collaborators"):
        template = st.selectbox("Report Template", REPORT_TEMPLATES)
        cols     = st.multiselect("Collaborators", options=[ADMIN_USER])
        if st.button("Send Invitations"):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(DB_PATH) as conn:
                for cusr in cols:
                    conn.execute(
                        "INSERT INTO collaborations(case_id,user,collaborator,timestamp) VALUES (?,?,?,?)",
                        (cid, user, cusr, ts)
                    )
                conn.commit()
            st.success("Invitations sent.")

    last = sqlite3.connect(DB_PATH).execute(
        "SELECT report_text FROM reports WHERE case_id=? ORDER BY timestamp DESC LIMIT 1", (cid,)
    ).fetchone()
    base = last[0] if last else ""
    txt  = st.text_area("Collaborative Report:", value=base, height=300)
    if st.button("Save Collaborative Report"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO reports(case_id,user,report_text,timestamp) VALUES (?,?,?,?)",
                (cid, user, f"[{template}] {txt}", ts)
            )
            conn.execute("UPDATE cases SET status=? WHERE case_id=?", ("in-review", cid))
            conn.commit()
        st.success("Report saved & in-review.")

# ----------------------------
# --- FOOTER
# ----------------------------
st.markdown("---")
st.caption("© 2025, Collaborative DICOM Review")