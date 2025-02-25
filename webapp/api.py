from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid,GridOptionsBuilder, JsCode
import base64
import requests

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

deathImage = f"data:image/svg+xml;base64,{image_to_base64('./images/death-skull-and-bones-svgrepo-com.svg')}"
aliveImage = f"data:image/svg+xml;base64,{image_to_base64('./images/heart-svgrepo-com.svg')}"

renderImage = JsCode("""
class ThumbnailRenderer {
    init(params) {

    this.eGui = document.createElement('img');
    this.eGui.setAttribute('src', params.value);
    this.eGui.setAttribute('width', 'auto');
    this.eGui.setAttribute('height', '100%');
    }
        getGui() {
        return this.eGui;
    }
}
 """)    

st.set_page_config(layout="wide")
st.write("""# Vas-tu mourir ?""")

with st.form(key='file_upload_form', clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload a file", type=["csv"])
    submit_button = st.form_submit_button(label='Prédiction')
    
    
if submit_button:

    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    pd_data = pd.read_csv(StringIO(bytes_data.decode('utf-8')))
    
    json_data = pd_data.replace({np.nan: None}).to_dict(orient='records')
    
    response = requests.post("http://localhost:8080/predict", json=json_data )
    res = response.json()
    
    # Add a column to the dataframe
    predictions = res.get('predictions', [])

    pd_data["result"] = [deathImage if x == 1 else aliveImage for x in predictions]
    
    gridOption = GridOptionsBuilder.from_dataframe(pd_data)
    gridOption.configure_column("result", cellRenderer=renderImage,headerName="Status",width=100,pinned="right")
    gridOptions = gridOption.build()
    AgGrid(pd_data, gridOptions=gridOptions, allow_unsafe_jscode=True)

    # Clear the file uploader
    uploaded_file = None