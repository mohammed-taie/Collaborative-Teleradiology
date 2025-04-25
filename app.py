import os
import io
import zipfile
import shutil
import sqlite3
import warnings
import pathlib
import json
from datetime import datetime

# Ensure NumPy is imported
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
# Corrected potential unicode space in DB_PATH definition if present
DB_PATH  = os.getenv("DB_PATH", "cases.db")

# Hardcoded simple credentials - WARNING: Insecure for production
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
    # Use 'with' statement for automatic connection closing
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code    TEXT,
            study_uid       TEXT UNIQUE,
            modality        TEXT,
            upload_date     TEXT,
            status          TEXT,
            assigned_to     TEXT,
            original_filename TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS comments (
            id          INTEGER PRIMARY KEY,
            case_id     INTEGER,
            user        TEXT,
            text        TEXT,
            timestamp   TEXT,
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
            id          INTEGER PRIMARY KEY,
            case_id     INTEGER,
            user        TEXT,
            shape_type  TEXT,
            shape_data  TEXT, -- Storing JSON as text
            timestamp   TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(case_id)
        );
        """)
        # No explicit commit needed with 'with' statement for executescript

init_db()

# ----------------------------
# --- UTILITY FUNCTIONS
# ----------------------------
def safe_extract(zip_bytes: io.BytesIO, extract_to: pathlib.Path):
    """Extract zip file, preventing Zip-Slip."""
    # Ensure extract_to is an absolute path for reliable comparison
    extract_to = extract_to.resolve()
    with zipfile.ZipFile(zip_bytes) as z:
        for member in z.infolist():
            # Skip directories
            if member.is_dir():
                continue
            # Construct the full potential path
            # Use sanitize_filename or similar if filenames can be malicious
            member_filename = pathlib.Path(member.filename).name # Flatten structure
            out_path = extract_to / member_filename
            # Resolve the potential output path
            abs_out_path = out_path.resolve()
            # Check if the resolved path is within the intended extraction directory
            if not str(abs_out_path).startswith(str(extract_to)):
                 raise RuntimeError(f"Attempted path traversal detected: {member.filename}")
        # If all paths are safe, extract
        z.extractall(path=extract_to)


def anonymize_and_save(ds: pydicom.Dataset, out_path: pathlib.Path):
    """Strip PHI and save DICOM safely."""
    ds.remove_private_tags()
    # List of tags to remove - consider using a more robust anonymizer profile if needed
    tags_to_remove = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientAge",
        "InstitutionName", "ReferringPhysicianName", "StudyDescription",
        "OperatorsName", "PatientAddress", "PatientTelephoneNumbers",
        # Add any other potentially identifying tags specific to your data source
    ]
    for tag in tags_to_remove:
        if tag in ds: # Use direct tag checking
            # Consider replacing with dummy values instead of deleting if needed for structure
            del ds[tag]
    # Save, ensuring parent directory exists might be needed if structure wasn't flat
    ds.save_as(str(out_path), write_like_original=False)

# Use st.cache_data for functions returning data (like numpy arrays)
@st.cache_data
def load_dicom_pixel_array(path: str): # Removed mtime, Streamlit handles caching based on args/code
    """Loads DICOM pixel array, cached by Streamlit."""
    try:
        ds = pydicom.dcmread(path)
        # Potential improvement: Apply rescale slope/intercept, VOI LUT if present
        # pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(ds.pixel_array, ds)
        # pixel_array = pydicom.pixel_data_handlers.util.apply_voi_lut(pixel_array, ds)
        return ds.pixel_array
    except Exception as e:
        st.error(f"Error reading DICOM file {pathlib.Path(path).name}: {e}")
        return None # Return None on error

# ----------------------------
# --- AUTHENTICATION
# ----------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None # Initialize user

if not st.session_state.authenticated:
    with st.sidebar:
        st.header("🔒 Login")
        # Use unique keys for inputs
        u = st.text_input("Username", key="login_user_input")
        p = st.text_input("Password", type="password", key="login_pass_input")
        if st.button("Login", key="login_button"):
            # TODO: Replace with secure hashing and user database lookup
            if u == ADMIN_USER and p == ADMIN_PASS:
                st.session_state.authenticated = True
                st.session_state.user = u
                st.rerun() # Rerun to update page state after login
            else:
                st.error("Invalid credentials")
    st.stop() # Stop execution if not authenticated

# Ensure user is set if authenticated state is somehow True without login flow
user = st.session_state.get("user", "Unknown") # Use .get for safety

with st.sidebar:
    st.markdown(f"**Logged in as:** {user}")
    if st.button("Logout", key="logout_button"):
        # Clear sensitive session state on logout
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun() # Rerun to go back to login state
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Upload Study", "Reporting & Collaboration"], key="nav_radio")
    st.markdown("---")
    with st.expander("Annotation Mode"):
        # Use unique keys for sidebar widgets
        annotate = st.checkbox("Enable Annotations", key="annotate_checkbox")
        shape    = st.selectbox("Shape", ["freedraw", "rect", "circle", "polygon"], key="shape_select")
        thickness= st.slider("Stroke width", 1, 10, 3, key="thickness_slider")

# ----------------------------
# --- UPLOAD STUDY PAGE
# ----------------------------
if page == "Upload Study":
    st.header("📤 Upload DICOM Study (.zip)")
    zip_file = st.file_uploader("ZIP of DICOM files", type="zip", key="zip_uploader")
    # Consider adding user selection for assignment if multiple users exist
    assigned_to = st.text_input("Assign to Radiologist", value=user, key="assign_input")

    if zip_file and st.button("Process Upload", key="upload_process_button"):
        fname = zip_file.name
        # Check if already uploaded based on filename
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                exists = conn.execute("SELECT 1 FROM cases WHERE original_filename=?", (fname,)).fetchone()
                if exists:
                    st.warning(f"File '{fname}' has already been uploaded.")
                    st.stop() # Stop processing this upload

            # Process the upload
            with st.spinner("Reading DICOM headers..."):
                # Read sample from memory without full extraction first
                with zipfile.ZipFile(zip_file) as zf:
                    # Find first file that isn't a directory and likely DICOM
                    dicom_member = None
                    for member in zf.infolist():
                         if not member.is_dir() and not member.filename.startswith('__MACOSX'):
                             # Basic check, could be more robust (e.g., check file extension)
                             dicom_member = member
                             break
                    if not dicom_member:
                         raise ValueError("No valid files found in the ZIP archive.")

                    sample_bytes = io.BytesIO(zf.read(dicom_member.filename))
                    sample_ds = pydicom.dcmread(sample_bytes, stop_before_pixels=True) # Read only header

                study_uid    = sample_ds.StudyInstanceUID
                patient_code = sample_ds.get("PatientID", f"Unknown_{study_uid[:8]}") # Use part of StudyUID if no PatientID
                modality     = sample_ds.get("Modality", "OT") # OT = Other
                upload_date  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status       = "new"

            # Check if Study Instance UID already exists
            with sqlite3.connect(DB_PATH) as conn:
                 conn.execute("PRAGMA foreign_keys = ON")
                 study_exists = conn.execute("SELECT 1 FROM cases WHERE study_uid=?", (study_uid,)).fetchone()
                 if study_exists:
                      st.warning(f"Study with UID '{study_uid}' has already been uploaded.")
                      st.stop()

            # Insert into database BEFORE file operations
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO cases
                        (patient_code, study_uid, modality, upload_date, status, assigned_to, original_filename)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (patient_code, study_uid, modality, upload_date, status, assigned_to, fname))
                case_id = cur.lastrowid # Get the ID for folder naming
                conn.commit() # Commit the transaction

            # Create folder and extract
            case_folder = pathlib.Path(DATA_DIR) / f"case_{case_id}"
            case_folder.mkdir(parents=True, exist_ok=True) # exist_ok handles race conditions

            with st.spinner(f"Extracting and anonymizing files for Case {case_id}..."):
                # Reset file pointer before extracting
                zip_file.seek(0)
                safe_extract(zip_file, case_folder)

                # Anonymize extracted files
                files_processed = 0
                files_failed = 0
                for f_path in case_folder.iterdir(): # Use iterdir
                    if f_path.is_file() and not f_path.name.startswith('.'): # Basic check for actual files
                        try:
                            # Use force=True to read files that might lack preamble/prefix
                            ds = pydicom.dcmread(str(f_path), force=True)
                            anonymize_and_save(ds, f_path)
                            files_processed += 1
                        except pydicom.errors.InvalidDicomError:
                            # Handle non-DICOM files gracefully (e.g., log, delete, or ignore)
                            print(f"Skipping non-DICOM or invalid file: {f_path.name}")
                            f_path.unlink() # Example: Delete non-DICOM files
                            files_failed += 1
                        except Exception as anon_err:
                            print(f"Error anonymizing file {f_path.name}: {anon_err}")
                            # Decide how to handle anonymization errors (e.g., delete file, mark case)
                            # f_path.unlink() # Option: delete problematic file
                            files_failed += 1

            # Clear cache for the specific function if needed, but @st.cache_data handles this well
            # load_dicom_pixel_array.clear() # Usually not needed with st.cache_data

            st.success(f"Upload complete! Case ID: {case_id}. Processed {files_processed} DICOM files. Skipped/Failed {files_failed} files.")
            # Consider removing st.rerun() if not strictly needed, allows user to see success message longer
            # st.rerun()

        except ValueError as ve:
             st.error(f"Upload Error: {ve}")
        except RuntimeError as rte:
             st.error(f"Security Error during extraction: {rte}")
        except sqlite3.IntegrityError as ie:
             # This might happen if UNIQUE constraint fails despite checks (race condition)
             st.error(f"Database Error: Potential duplicate entry. Please check if the case already exists. ({ie})")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # Clean up potential partial data? (e.g., delete case folder, DB entry) - depends on desired atomicity

# ----------------------------
# --- DASHBOARD PAGE
# ----------------------------
elif page == "Dashboard":
    st.title("📁 Case Dashboard")
    with st.sidebar.expander("Filters"):
        # Use unique keys
        status_filter = st.selectbox("Status", ["All", "new", "in-review", "finalized"], key="status_filter_select")
        my_only       = st.checkbox("My cases only", value=True, key="my_cases_checkbox")

    # Build query safely
    query   = "SELECT case_id, patient_code, modality, upload_date, status, assigned_to FROM cases"
    params  = []
    clauses = []
    if status_filter != "All":
        clauses.append("status = ?"); params.append(status_filter)
    if my_only:
        # Ensure 'user' variable is correctly scoped and assigned from session_state
        if 'user' in st.session_state and st.session_state.user:
             clauses.append("assigned_to = ?"); params.append(st.session_state.user)
        else:
             # Handle case where user somehow isn't set - maybe show no cases?
             st.warning("User not identified, cannot filter 'My cases only'.")
             clauses.append("1 = 0") # Effectively show no results

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY upload_date DESC"

    # Fetch data
    try:
        with sqlite3.connect(DB_PATH) as conn:
             cases = conn.execute(query, params).fetchall()
    except Exception as db_e:
        st.error(f"Database error fetching cases: {db_e}")
        cases = [] # Ensure cases is empty list on error

    if not cases:
        st.info("No cases found matching the current filters.")
    else:
        for cid, pc, md, ud, stt, asgn in cases:
            # Use case ID in expander key for uniqueness
            expander_key = f"expander_case_{cid}"
            with st.expander(f"Case {cid} | Patient: {pc} | Modality: {md} | Status: {stt} | Assigned: {asgn}", key=expander_key):
                folder = pathlib.Path(DATA_DIR) / f"case_{cid}"
                # Filter for likely DICOM extensions, case-insensitive
                dcms = sorted([p for p in folder.glob("*") if p.suffix.lower() in ['.dcm', '.dicom', '']]) # Include files with no extension
                # Filter out directories just in case glob picked them up
                dcms = [p for p in dcms if p.is_file()]

                if not dcms:
                    st.warning("No DICOM files found in this case folder.")
                    continue # Skip to next case

                col1, col2 = st.columns([3, 1])
                with col1:
                    slice_count = len(dcms)
                    # Only show slider if more than one slice
                    if slice_count > 1:
                         idx = st.slider("Slice", 0, slice_count - 1, 0, key=f"slice_slider_{cid}")
                    elif slice_count == 1:
                         idx = 0
                         st.write("Single slice study.") # Indicate single slice
                    else:
                         # This case should be caught by 'if not dcms' above, but as a safeguard:
                         st.warning("No slices available to display.")
                         continue

                    # Load pixel array using the cached function
                    selected_dcm_path = str(dcms[idx])
                    arr = load_dicom_pixel_array(selected_dcm_path)

                    # Handle potential loading errors from the cached function
                    if arr is None:
                         st.error(f"Could not load pixel data for slice {idx}.")
                         continue # Skip rendering for this slice

                    # --- ANNOTATION / PLOTLY DISPLAY ---
                    if annotate:
                        # Calculate scaling and dimensions
                        try:
                            scale = min(1.0, CANVAS_MAX_WIDTH / arr.shape[1]) if arr.shape[1] > 0 else 1.0
                            w = int(arr.shape[1] * scale)
                            h = int(arr.shape[0] * scale)

                            # Prepare background image - ensure correct mode for canvas
                            # Normalize array if necessary (e.g., scale to 0-255 for uint8)
                            # This step is crucial if DICOM data isn't standard uint8/uint16 range
                            if arr.dtype != np.uint8:
                                 # Basic normalization - might need adjustment based on modality/data range
                                 norm_arr = arr.astype(np.float32)
                                 min_val, max_val = np.min(norm_arr), np.max(norm_arr)
                                 if max_val > min_val:
                                      norm_arr = 255 * (norm_arr - min_val) / (max_val - min_val)
                                 norm_arr = norm_arr.astype(np.uint8)
                            else:
                                 norm_arr = arr

                            bg = Image.fromarray(norm_arr).resize((w, h))
                            # Ensure image mode is compatible (e.g., RGB/RGBA often safer)
                            if bg.mode not in ['RGB', 'RGBA']:
                                 bg = bg.convert('RGB') # Convert grayscale 'L' or other modes

                            # --- Debug Prints (Check Logs on Streamlit Cloud) ---
                            print(f"--- Debug Info for Case {cid}, Slice {idx} ---")
                            print(f"Original array type: {type(arr)}, dtype: {arr.dtype}, shape: {arr.shape}")
                            try:
                                print(f"Original array min: {np.min(arr)}, max: {np.max(arr)}")
                            except Exception as e_minmax:
                                print(f"Could not get min/max: {e_minmax}")
                            print(f"Background image type: {type(bg)}")
                            print(f"Background image mode: {bg.mode}")
                            print(f"Background image size: {bg.size}")
                            print(f"--- End Debug Info ---")
                            # --- End Debug Prints ---

                            # --- Try-Except block for st_canvas ---
                            try:
                                canvas = st_canvas(
                                    fill_color="rgba(0,0,0,0)",
                                    stroke_width=thickness,
                                    stroke_color="#FF0000",
                                    background_image=bg,
                                    # update_streamlit=True, # Consider for real-time feedback if needed
                                    drawing_mode=shape,
                                    key=f"canvas_{cid}_{idx}", # Ensure unique key
                                    width=w,
                                    height=h
                                )

                                # Save Annotation Logic (only if canvas loaded successfully)
                                if st.button("Save Annotation", key=f"save_annot_button_{cid}_{idx}"):
                                     if canvas.json_data and canvas.json_data.get("objects"):
                                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        inv_scale = 1.0 / scale if scale > 0 else 1.0 # Avoid division by zero

                                        annotations_to_save = []
                                        for obj_data in canvas.json_data["objects"]:
                                            # Create a deep copy to avoid modifying original canvas data
                                            obj = json.loads(json.dumps(obj_data))
                                            obj_type = obj.get("type")

                                            # Scale coordinates back to original image size
                                            if obj_type in ("rect", "circle", "path"): # Path for freedraw
                                                if 'left' in obj: obj["left"] *= inv_scale
                                                if 'top' in obj: obj["top"] *= inv_scale
                                                if obj_type == "rect":
                                                    if 'width' in obj: obj["width"] *= inv_scale
                                                    if 'height' in obj: obj["height"] *= inv_scale
                                                elif obj_type == "circle":
                                                    if 'radius' in obj: obj["radius"] *= inv_scale
                                                # Freedraw paths need scaling too
                                                elif obj_type == "path" and 'path' in obj:
                                                     for point_cmd in obj['path']:
                                                          if len(point_cmd) > 1: point_cmd[1] *= inv_scale
                                                          if len(point_cmd) > 2: point_cmd[2] *= inv_scale
                                                # Add other types if needed (e.g., polygon points)

                                            # Append tuple for executemany
                                            annotations_to_save.append(
                                                (cid, user, obj_type, json.dumps(obj), ts)
                                            )
                                            print(f"Prepared annotation: {obj_type}") # Log prepared annotation

                                        if annotations_to_save:
                                            try:
                                                with sqlite3.connect(DB_PATH) as conn:
                                                    conn.executemany(
                                                        "INSERT INTO annotations(case_id, user, shape_type, shape_data, timestamp) VALUES (?,?,?,?,?)",
                                                        annotations_to_save
                                                    )
                                                    conn.commit()
                                                st.success(f"Saved {len(annotations_to_save)} annotation(s)!")
                                            except Exception as db_annot_err:
                                                st.error(f"Database error saving annotations: {db_annot_err}")
                                        else:
                                             st.warning("No annotations drawn to save.")
                                     else:
                                        st.warning("No annotation data received from canvas.")


                            # Catch the specific error we are debugging
                            except AttributeError as ae:
                                print(f"!!! AttributeError occurred in st_canvas for case {cid}, slice {idx}: {ae}")
                                st.error(f"⚠️ Failed to load annotation canvas for this image. Error: {ae}", icon="🔥")
                                st.warning("Annotation is disabled for this slice due to an error.")
                            except Exception as e_canvas:
                                print(f"!!! Generic error occurred in st_canvas for case {cid}, slice {idx}: {e_canvas}")
                                st.error(f"⚠️ An unexpected error occurred while loading the annotation canvas: {e_canvas}", icon="🔥")
                                st.warning("Annotation is disabled for this slice due to an error.")

                        except Exception as e_img_prep:
                             print(f"!!! Error preparing image for canvas (case {cid}, slice {idx}): {e_img_prep}")
                             st.error(f"Could not prepare image for annotation: {e_img_prep}")

                    else: # If not annotate
                        # Display using Plotly
                        try:
                             # Normalize for consistent display if not uint8
                             if arr.dtype != np.uint8:
                                  norm_arr = arr.astype(np.float32)
                                  min_val, max_val = np.min(norm_arr), np.max(norm_arr)
                                  if max_val > min_val:
                                      norm_arr = (norm_arr - min_val) / (max_val - min_val)
                                  # Plotly imshow often handles float 0-1 range well
                             else:
                                  norm_arr = arr

                             fig = px.imshow(norm_arr, color_continuous_scale="gray", aspect="equal")
                             fig.update_layout(
                                 dragmode="pan",
                                 margin=dict(l=0, r=0, t=0, b=0),
                                 coloraxis_showscale=False # Hide color scale bar
                             )
                             fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                             fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                             st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
                        except Exception as e_plotly:
                             st.error(f"Could not display image using Plotly: {e_plotly}")

                # --- Right Column (Comments & Report) ---
                with col2:
                    # Comments Section
                    st.subheader("💬 Comments")
                    comment_text = st.text_area("Add comment:", key=f"comment_input_{cid}", height=100)
                    if st.button("Post Comment", key=f"post_comment_button_{cid}"):
                        if comment_text:
                             ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                             try:
                                 with sqlite3.connect(DB_PATH) as conn:
                                      conn.execute(
                                           "INSERT INTO comments(case_id, user, text, timestamp) VALUES (?,?,?,?)",
                                           (cid, user, comment_text, ts)
                                      )
                                      conn.commit()
                                 st.success("Comment added.")
                                 # Rerun might be needed to clear the text_area and show the new comment immediately
                                 st.rerun()
                             except Exception as db_comment_err:
                                 st.error(f"Database error posting comment: {db_comment_err}")
                        else:
                             st.warning("Comment cannot be empty.")

                    # Display existing comments
                    st.markdown("---")
                    try:
                         with sqlite3.connect(DB_PATH) as conn:
                              comments = conn.execute(
                                   "SELECT user, text, timestamp FROM comments WHERE case_id=? ORDER BY timestamp DESC", (cid,)
                              ).fetchall()
                         if comments:
                              # Limit number of displayed comments? Add scrolling?
                              with st.container(height=200): # Make comments scrollable
                                  for c_user, c_text, c_ts in comments:
                                      st.markdown(f"**{c_user}** ({c_ts}):\n> {c_text}")
                         else:
                              st.caption("No comments yet.")
                    except Exception as db_fetch_comment_err:
                         st.error(f"Could not load comments: {db_fetch_comment_err}")

                    # Report Section
                    st.subheader("📄 Report")
                    try:
                         with sqlite3.connect(DB_PATH) as conn:
                              last_report_result = conn.execute(
                                   "SELECT report_text FROM reports WHERE case_id=? ORDER BY timestamp DESC LIMIT 1", (cid,)
                              ).fetchone()
                         base_report = last_report_result[0] if last_report_result else ""
                    except Exception as db_fetch_report_err:
                         st.error(f"Could not load last report: {db_fetch_report_err}")
                         base_report = ""

                    report_text = st.text_area("Findings:", value=base_report, key=f"report_input_{cid}", height=200)
                    if st.button("Save Report & Finalize", key=f"save_report_button_{cid}"):
                         if report_text:
                              ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                              try:
                                   with sqlite3.connect(DB_PATH) as conn:
                                        # Insert new report version
                                        conn.execute(
                                             "INSERT INTO reports(case_id, user, report_text, timestamp) VALUES (?,?,?,?)",
                                             (cid, user, report_text, ts)
                                        )
                                        # Update case status
                                        conn.execute("UPDATE cases SET status=? WHERE case_id=?", ("finalized", cid))
                                        conn.commit()
                                   st.success("Report saved & case finalized.")
                                   # Rerun to update status display
                                   st.rerun()
                              except Exception as db_save_report_err:
                                   st.error(f"Database error saving report: {db_save_report_err}")
                         else:
                              st.warning("Report text cannot be empty.")


# ----------------------------
# --- REPORTING & COLLABORATION PAGE
# ----------------------------
elif page == "Reporting & Collaboration":
    st.title("🤝 Reporting & Collaboration")
    try:
         with sqlite3.connect(DB_PATH) as conn:
              cases = conn.execute(
                   "SELECT case_id, patient_code, modality, status FROM cases ORDER BY upload_date DESC"
              ).fetchall()
    except Exception as db_e:
        st.error(f"Database error fetching cases: {db_e}")
        cases = []

    if not cases:
        st.warning("No cases available for reporting.")
        st.stop()

    # Create options for selectbox: "Case ID | Patient Code | Modality | Status"
    opts = {f"{c[0]} | {c[1]} | {c[2]} | {c[3]}": c[0] for c in cases}
    # Use a unique key for the selectbox
    selected_case_key = st.selectbox("Choose Case", list(opts.keys()), key="collab_case_select")
    # Handle case where opts is empty (should be caught above but safety)
    selected_case_id = opts.get(selected_case_key) if selected_case_key else None

    if selected_case_id is None:
         st.error("Could not determine selected case.")
         st.stop()

    st.subheader(f"Editing Report for Case {selected_case_id}")

    # --- Sidebar: Collaboration ---
    with st.sidebar.expander("Collaboration Settings"):
         # Use unique keys
         template = st.selectbox("Report Template", REPORT_TEMPLATES, key="template_select")
         # TODO: Replace [ADMIN_USER] with a dynamic list of actual users from a User table
         collaborators = st.multiselect("Collaborators", options=[ADMIN_USER], key="collaborator_multiselect")
         if st.button("Send Invitations", key="invite_button"): # Currently just saves to DB
              ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              invites_to_add = []
              for collab_user in collaborators:
                   # Avoid inviting self? Check if already invited?
                   invites_to_add.append((selected_case_id, user, collab_user, ts))

              if invites_to_add:
                   try:
                       with sqlite3.connect(DB_PATH) as conn:
                            # Add logic to prevent duplicate invites if needed
                            conn.executemany(
                                 "INSERT INTO collaborations(case_id, user, collaborator, timestamp) VALUES (?,?,?,?)",
                                 invites_to_add
                            )
                            conn.commit()
                       st.success(f"Collaboration entries added for {len(invites_to_add)} user(s).")
                   except Exception as db_collab_err:
                       st.error(f"Database error saving collaborations: {db_collab_err}")
              else:
                  st.warning("No collaborators selected.")

    # --- Main Area: Report Editing ---
    try:
         with sqlite3.connect(DB_PATH) as conn:
              last_report_result = conn.execute(
                   "SELECT report_text FROM reports WHERE case_id=? ORDER BY timestamp DESC LIMIT 1", (selected_case_id,)
              ).fetchone()
         base_report = last_report_result[0] if last_report_result else ""
    except Exception as db_fetch_report_err:
         st.error(f"Could not load last report for case {selected_case_id}: {db_fetch_report_err}")
         base_report = ""

    # Use selected template to potentially pre-fill or structure the report
    # Simple example: prepend template name if base_report is empty
    if not base_report and template != "Free-Text":
         initial_report_text = f"[{template}]\n\nFindings:\n\nImpression:\n"
    else:
         initial_report_text = base_report

    # Use unique key for text area
    report_text = st.text_area("Collaborative Report:", value=initial_report_text, height=400, key=f"collab_report_input_{selected_case_id}")

    # Button to save (and potentially change status)
    if st.button("Save Collaborative Report", key="save_collab_report_button"):
         if report_text:
              ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              # Add template marker if not already present? Or just save text as is?
              final_report_text = report_text # Modify as needed, e.g., add template marker if desired

              try:
                   with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                             "INSERT INTO reports(case_id, user, report_text, timestamp) VALUES (?,?,?,?)",
                             (selected_case_id, user, final_report_text, ts)
                        )
                        # Update status to 'in-review' when a collaborative report is saved
                        conn.execute("UPDATE cases SET status=? WHERE case_id=?", ("in-review", selected_case_id))
                        conn.commit()
                   st.success("Collaborative report saved. Case status set to 'in-review'.")
                   # Consider rerun to show updated status in dropdown?
                   # st.rerun()
              except Exception as db_save_collab_err:
                   st.error(f"Database error saving collaborative report: {db_save_collab_err}")
         else:
              st.warning("Report text cannot be empty.")


# ----------------------------
# --- FOOTER
# ----------------------------
st.markdown("---")
# Update year dynamically or remove if preferred
current_year = datetime.now().year
st.caption(f"© {current_year}, Collaborative DICOM Review")