
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import streamlit as st
from utils import extract_nutrients, show_grid
from fatsecret import Fatsecret

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

st.header(':cake: Calorie Counter')
with st.expander("ℹ️ About this app", expanded=False):

    st.write(
        """     
-   The *Calorie Counter* app is a demo program built in Streamlit for analyzing nutrition factors.
-   It uses the [*FatSecret Platform API*](https://platform.fatsecret.com/api/Default.aspx?screen=home) to analyze nutrients from input text.
	    """
    )
    img = Image.open('diagram.png')
    st.image(img)

st.markdown("")

fs = Fatsecret('4efcbe23669243e98086f59e404ef125', '3bc000b4b56046c3b6756022017f8aa3')


menu_list = list()
selected_foods = pd.DataFrame()
if "mdf" not in st.session_state:
    st.session_state.mdf = pd.DataFrame(columns=['food_id', 'food_name', 'begin_dtm', 'end_dtm', 'amount', 'total_carb', 'total_fat',
                                                 'total_protein', 'total_calories'])

menu = st.text_input('Enter food name', help='검색할 음식 이름을 한글로 입력하고, Enter키를 입력하거나 `Search` 버튼을 클릭하세요.')

if st.button('Search', key = 'menu') | bool(menu):
    if menu.strip() !='':
        foods = pd.DataFrame()
        try:
            foods= pd.DataFrame(fs.foods_search(search_expression=menu, region='KR'))
        except:
            st.warning('Enter food name')

        if foods.shape[0] > 0:
            foods[['serving_unit', 'base_amount', 'unit', 'kcal', 'fat', 'carb', 'protein']] = \
                foods.apply(extract_nutrients, axis=1, result_type='expand')

            if 'brand_name' not in foods.columns:
                foods = foods[['food_id', 'food_name', 'food_type',
                                    'serving_unit', 'base_amount', 'unit', 'kcal', 'fat', 'carb', 'protein']]
            else:
                foods = foods[['food_id', 'food_name', 'food_type', 'brand_name',
                                'serving_unit', 'base_amount', 'unit', 'kcal', 'fat', 'carb', 'protein']]

            foods[['base_amount', 'kcal', 'fat', 'carb', 'protein']] = \
                foods[['base_amount', 'kcal', 'fat', 'carb', 'protein']].astype('float32')

            gb = GridOptionsBuilder.from_dataframe(foods, min_column_width=10)
            # enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
            gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True, editable=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
            gridOptions = gb.build()

            response = AgGrid(
                foods,
                height=250,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )
            selected_foods = pd.DataFrame(response["selected_rows"])
            #st.table(selected_foods)

            if response["selected_rows"]:
                menu_list = selected_foods['food_name'].to_list()
                menu_list = st.multiselect('Selected food(s)', menu_list, menu_list, help='저장할 음식을 선택하세요.')
                # st.write('selected food(s)')
                # menu_list = selected_foods['food_name'].to_list()
                # text = ''
                # for e in menu_list:
                #     text += "- " + e + "\n"
                # st.markdown(text)

st.markdown("")
with st.form(key="my_form", clear_on_submit = True):
    c21, c22, c23 = st.columns([1, 0.02, 1])

    with c21:
        today = st.date_input('Date', datetime.datetime.now(), help='식사한 날짜를 입력하세요.')
        meal_time = st.number_input('Meal time(min)', value=15, min_value=5, max_value=60, step=5,
                                    help = '식사를 지속한 시간을 5분 단위로 입력하세요.')
        submit = st.form_submit_button(label=' Add ')


    with c23:
        time = st.time_input('Time', datetime.time(), help='식사를 시작한 시간을 입력하세요.')

        # start = "00:00"
        # end = "23:55"
        # times = []
        # start = now = datetime.datetime.strptime(start, "%H:%M")
        # end = datetime.datetime.strptime(end, "%H:%M")
        # while now != end:
        #     times.append(str(now.strftime("%H:%M")))
        #     now += datetime.timedelta(minutes=5)
        # times.append(end.strftime("%H:%M"))
        # time = st.multiselect('Time', times, help= '식사를 시작한 시간을 입력하세요.')
        # time = datetime.datetime.strptime(time, "%H:%M")
        amount = st.selectbox('Amount', ['1/4', '1/2', '3/4', '1', '1+1/2', '2'], help='식사량을 입력하세요.')
        map = {'1/4':0.25, '1/2':0.5, '3/4':0.75, '1':1.0, '1+1/2':1.5, '2':2.0}
        amount_factor = map[amount]
        #remove = st.form_submit_button(label='remove', on_click=update, args=[grid_table])

    if (selected_foods.shape[0] > 0) & submit:
        df_new = pd.DataFrame(columns=['food_id', 'food_name', 'begin_dtm', 'amount', 'total_carb', 'total_fat',
                                       'total_protein', 'total_calories'])
        for item in menu_list:
            df_new = pd.DataFrame({'food_id': selected_foods[selected_foods['food_name'] == item]['food_id'].values[0],
                                   'food_name': selected_foods[selected_foods['food_name'] == item]['food_name'].values[
                                       0],
                                   'begin_dtm': datetime.datetime.combine(today, time),
                                   'end_dtm': datetime.datetime.combine(today, time) + datetime.timedelta(
                                       minutes=meal_time),
                                   'amount': amount_factor,
                                   'total_carb': np.round(amount_factor * selected_foods[selected_foods['food_name'] == item]['carb'].values[0], 2),
                                   'total_fat': np.round(amount_factor * selected_foods[selected_foods['food_name'] == item]['fat'].values[0], 2),
                                   'total_protein': np.round(amount_factor * selected_foods[selected_foods['food_name'] == item]['protein'].values[0], 2),
                                   'total_calories': np.round(amount_factor *selected_foods[selected_foods['food_name'] == item]['kcal'].values[0], 2),
                                   }, index = [0])
            st.session_state.mdf = pd.concat([st.session_state.mdf, df_new], axis=0).reset_index(drop = True)


    if st.session_state.mdf.shape[0]>0:
        grid_table = show_grid(st.session_state.mdf)


