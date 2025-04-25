# 🏥 AI DICOM Review

A Streamlit-based web app for secure, collaborative DICOM study review and reporting. Upload ZIP archives of DICOM series, anonymize & view images, annotate regions of interest, comment, and generate structured reports. Designed for quick prototyping on Streamlit Community Cloud, with easy migration to a more scalable backend.

---

## 🚀 Features

- **Secure Upload & Extraction**  
  - Prevents zip-slip attacks  
  - Strips private DICOM tags (PHI)  
- **Interactive Dashboard**  
  - Browse all cases by status (new / in-review / finalized)  
  - Slice slider for multi-slice studies  
  - Soft pan/zoom via Plotly  
- **Annotation Mode**  
  - Free-draw, rectangle, circle, polygon  
  - Stores vector shapes in SQLite  
- **Collaboration & Reporting**  
  - Comment thread per case  
  - Save free-text, BI-RADS, PI-RADS reports  
  - Invite collaborators (via simple invite log)  
- **Easy Deployment**  
  - One-file `app.py`  
  - Configured for Streamlit Community Cloud  
  - Environment & secrets via `st.secrets.toml`

---

## 🗂️ Repository Structure