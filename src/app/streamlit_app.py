"""Streamlit demo — 4 tabs. Implemented by O4."""
import streamlit as st

st.set_page_config(page_title="BloodMNIST Classifier", layout="wide")

st.title("BloodMNIST Classifier")
st.info(
    "Placeholder aplikacji. O4: Predykcja | Metryki | Wyjaśnialność | O projekcie. "
    "Dataset nie jest przeznaczony do użycia klinicznego."
)

tab_pred, tab_metrics, tab_xai, tab_about = st.tabs(
    ["Predykcja", "Metryki", "Wyjaśnialność", "O projekcie"]
)

with tab_pred:
    st.write("Wybór obrazu z testu / upload — po wytrenowaniu modeli (O3/O4).")

with tab_metrics:
    st.write("Wczytanie `reports/results/*.json` — po ewaluacji (O2/O3).")

with tab_xai:
    st.write("Grad-CAM — po implementacji (O4).")

with tab_about:
    st.markdown(
        """
        **BloodMNIST** (MedMNIST v2) — klasyfikacja 8 typów krwinek, 28×28 RGB.

        Zespół: O1 Data | O2 Classical ML | O3 Deep Learning | O4 XAI & integracja.

        Cytowania: Yang et al., Scientific Data 2023; Acevedo et al.
        """
    )
