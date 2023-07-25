import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode
import streamlit as st

def extract_nutrients(text):
    text = text['food_description']  # multiple output apply 함수 구현을 위해 추가함
    item_splited = text.split('|')
    fat = item_splited[1].replace('지방:', '').replace(',', '').replace('g', '').strip()
    carb = item_splited[2].replace('탄수화물:', '').replace(',', '').replace('g', '').strip()
    protein = item_splited[3].replace('단백질:', '').replace(',', '').replace('g', '').strip()
    kcal_splited = item_splited[0].split('당')
    kcal = kcal_splited[1].replace('- 칼로리:', '').replace(',', '').replace('kcal', '').strip()

    base_amount_splited = kcal_splited[0].split('(')

    if len(base_amount_splited) == 1:  # serving unit가 없는 경우
        base_amount = ''.join(filter(lambda i: i.isdigit(), base_amount_splited[0]))
        unit = base_amount_splited[0].replace(base_amount, '').replace(')', '').strip()  # 단위만 추출

        return [np.nan, base_amount, unit, kcal, fat, carb, protein]

    else:
        serving_unit = base_amount_splited[0]
        base_amount = ''.join(filter(lambda i: i.isdigit(), base_amount_splited[1]))
        unit = base_amount_splited[1].replace(base_amount, '').replace(')', '').strip()  # 단위만 추출

        return [serving_unit, base_amount, unit, kcal, fat, carb, protein]

def show_grid(df):
    gb = GridOptionsBuilder.from_dataframe(df, min_column_width=10)

    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
    gridOptions = gb.build()

    grid_table = AgGrid(
                df,
                height = 250,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )
    return grid_table

def update(grid_table):
    grid_table_df = pd.DataFrame(grid_table['data'])
    selected_rows = pd.DataFrame(grid_table['selected_rows'])
    result_df = pd.concat([st.session_state.mdf, pd.DataFrame(st.session_state.mdf_result)])
    result_df = result_df[~result_df.duplicated(keep=False)]

    return show_grid(result_df)