from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import base64
import requests

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Convertir les images en Base64
deathImage = f"data:image/svg+xml;base64,{image_to_base64('./images/death-skull-and-bones-svgrepo-com.svg')}"
aliveImage = f"data:image/svg+xml;base64,{image_to_base64('./images/heart-svgrepo-com.svg')}"

st.set_page_config(layout="wide")
st.title("Vas-tu mourir ?")

if 'pd_data' not in st.session_state:
    st.session_state.pd_data = None

with st.form(key='file_upload_form', clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])
    submit_button = st.form_submit_button(label='Prédiction')

if submit_button and uploaded_file is not None:
    bytes_data = uploaded_file.read()
    pd_data = pd.read_csv(StringIO(bytes_data.decode('utf-8')))
    
    json_data = pd_data.replace({np.nan: None}).to_dict(orient='records')
    response = requests.post("http://localhost:8080/predict", json=json_data)
    res = response.json()
    
    # Ajout des prédictions sous forme d'images
    predictions = res.get('predictions', [])
    pd_data["prediction"] = predictions
    pd_data["target"] = [None] * len(pd_data)

    st.session_state.pd_data = pd_data

if st.session_state.pd_data is not None:
    pd_data = st.session_state.pd_data
    st.write("### Résultats des prédictions")

    # Affichage ligne par ligne sous forme de tableau
    for index, row in pd_data.iterrows():
        st.write(f"#### Ligne {index + 1}")

        # Transformation de la ligne en dataframe pour affichage dans `st.table`
        row_df = pd.DataFrame(row).transpose()
        row_df = row_df.drop(columns=["prediction","target"])  # Supprimer les colonnes image et prediction pour éviter un bug d'affichage

        # Affichage du tableau
        st.table(row_df)

        st.write(f"##### Prediction")

        # Affichage de l'image du résultat et du label de prédiction
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            st.image( deathImage if row["prediction"] == 1 else aliveImage, width=50)
        
        st.write(f"##### Feedback")

        if st.session_state.pd_data.at[index, "target"] is not None:
            st.image(deathImage if st.session_state.pd_data.at[index, "target"] == 1 else aliveImage, width=50)
        else:
        # Boutons de feedback avec images
            col4, col5 = st.columns([1, 1])
            with col4:
                if st.button("💀", key=f"death_{index}"):
                    st.session_state.pd_data.at[index, "target"] = 1
                    data = pd_data.iloc[index].replace({np.nan: None}).to_dict()
                    print(data)
                    requests.post("http://localhost:8080/feedback", json=[data])
                    st.rerun()

            with col5:
                if st.button("❤️", key=f"alive_{index}"):
                    st.session_state.pd_data.at[index, "target"] = 0
                    data = pd_data.iloc[index].replace({np.nan: None}).to_dict()
                    print(data)
                    
                    requests.post("http://localhost:8080/feedback", json=[data])
                    st.rerun()

        st.write("---")  # Séparateur entre chaque ligne