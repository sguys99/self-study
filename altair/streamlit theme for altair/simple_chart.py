import streamlit as st

import altair as alt
import pandas as pd
import numpy as np

from vega_datasets import data

source = data.movies()

st.dataframe(source.head())

chart = alt.Chart(source).mark_bar().encode(
    alt.X("IMDB_Rating:Q", bin=True),
        y='count()',
    )


tab1, tab2 = st.tabs(["Streamlit 테마 (default)", "Altair native theme"])

with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
with tab2:
    st.altair_chart(chart, theme=None, use_container_width=True)
    

x = np.arange(100)
source = pd.DataFrame({
    'x': x,
    'f(x)': np.sin(x / 5)
})

chart = alt.Chart(source).mark_line().encode(
    x='x',
    y='f(x)'
    )

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
with tab2:
    st.altair_chart(chart, theme=None, use_container_width=True)
    
    
source = data.cars()
st.dataframe(source.head())

chart = alt.Chart(source).mark_circle(size=60).encode(
        x='Horsepower',
        y='Miles_per_Gallon',
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
with tab2:
    st.altair_chart(chart, theme=None, use_container_width=True)