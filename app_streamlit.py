"""Demo visual opcional. Requiere la API corriendo en el puerto 8000.

streamlit run app_streamlit.py
"""
import os

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
KEY = os.getenv("APP_API_KEY", "demo-key")
H = {"X-API-Key": KEY}

st.set_page_config(page_title="Asistente de EPP", layout="wide")
st.title("Asistente de cumplimiento de EPP")

tab1, tab2 = st.tabs(["Inspección de imagen", "Consulta normativa"])

with tab1:
    zonas = httpx.get(f"{API}/zonas", headers=H, timeout=30).json()
    zona = st.selectbox("Zona", list(zonas), format_func=lambda z: zonas[z]["nombre"])
    img = st.file_uploader("Imagen (jpg o png)", type=["jpg", "jpeg", "png"])
    if img and st.button("Inspeccionar"):
        r = httpx.post(f"{API}/inspect", headers=H, data={"zona": zona},
                       files={"imagen": (img.name, img.getvalue(), img.type)}, timeout=120)
        out = r.json()
        st.image(img, width=320)
        if r.status_code != 200:
            st.error(out.get("detail"))
        else:
            (st.success if out["cumple"] else st.error)(out["dictamen"])
            st.write("Estado por elemento:", out["estado"])
            st.write("Fuentes:", out["fuentes"], "· Rondas de reflexión:", out["rondas_reflexion"])
            if out.get("estado_alerta") == "pendiente_aprobacion":
                st.session_state["pendiente"] = out["run_id"]
            with st.expander("Traza entre agentes"):
                st.json(out["traza"])

    if st.session_state.get("pendiente"):
        st.warning("Revisión humana: ¿se envía la alerta de incumplimiento?")
        c1, c2 = st.columns(2)
        for col, label, ok in ((c1, "Aprobar y enviar", True), (c2, "Rechazar", False)):
            if col.button(label):
                r = httpx.post(f"{API}/inspect/{st.session_state['pendiente']}/decision", headers=H,
                               json={"aprobar": ok}, timeout=60)
                st.info(f"Estado de la alerta: {r.json().get('estado_alerta')}")
                st.session_state["pendiente"] = None

with tab2:
    q = st.text_input("Pregunta", "¿Qué EPP es obligatorio en el depósito?")
    if st.button("Consultar"):
        r = httpx.post(f"{API}/ask", headers=H, json={"pregunta": q}, timeout=120)
        out = r.json()
        if r.status_code != 200:
            st.error(out.get("detail"))
        else:
            st.write(out["respuesta"])
            st.caption(f"Fuentes: {', '.join(out['fuentes'])}")
