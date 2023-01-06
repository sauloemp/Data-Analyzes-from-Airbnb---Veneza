
import numpy as np
import plotly.express as px


#streamlit Libries
import streamlit as st
import streamlit.components.v1 as components
#Vizualizations libries
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,8)
import plotly.express as px
import plotly.graph_objects as go


#sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

#Configurations
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)



# Data Reading
df = pd.read_csv('http://data.insideairbnb.com/italy/veneto/venice/2022-09-07/visualisations/listings.csv', encoding = 'utf8')

# Feature Selection and ETL part
# In this step and to create a model, I'll conside Just Numeric fields, that I belive impact in price.

#Data Transformation
df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')

#Put a feature to identify if there is or there isn't a contract
df['there_is_licence'] = df['license'].apply(lambda x: 1 if type(x) == str else 0)
df['diff_days'] = round((pd.to_datetime("now") - df['last_review'])/ np.timedelta64(1, 'D'),0)
#Features Selection 
df_model_base = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm', 'there_is_licence', 'diff_days']]
df_model_base= df_model_base.fillna(df_model_base.mean())

Neighbourhood_choices = list(df["neighbourhood"].unique())




st.set_page_config(
    page_title = "Airbnb Solution Price",
    page_icon="Image\icon.png",
    layout = "wide",
    menu_items = {
        'Get Help': 'https://github.com/sauloemp',
        'Report a bug': 'https://github.com/sauloemp',
        'About': "It's a personal chalange working with data from airbnb Veneza"
    }
)



with st.sidebar:
    st.header("Data Analyzes from Airbnb - Veneza")
    st.markdown(
    '''
    # Introduction
    This work is based in [raffaelhfarias](https://github.com/raffaelhfarias/Dados_Airbnb) work, I saw his analyes and I would like to contribuilt with some analyzes. In view of this, I strong advice everyone to see his work before and come back here later.

    In this notebook, I'll consider the aspects from this dataset: [Veneza Airbnb Data](http://data.insideairbnb.com/italy/veneto/venice/2022-09-07/visualisations/listings.csv)

    [raffaelhfarias](https://github.com/raffaelhfarias/Dados_Airbnb) already developed a data dictionary.


    ## Who am I...
    I'm Saulo. I studied with Rafael and his project call my attention. Because when I was started in data science area I didn't had anyone could give me any kind of way, and I decided help him to show my POV about his project. And become his project more powerful. And maybe one day merge both projects in one.

    if do you want know more about me, follow my contacts: [LinkeIn](https://www.linkedin.com/in/saulo-pereira/) and [Github](https://github.com/sauloemp)

    ''')
    

def Compare(df,Neigbourhood1, Neigbourhood2):
  if Neigbourhood1 != Neigbourhood2:
    Main_n1_price = round(df[df['neighbourhood']== Neigbourhood1]['price'].dropna().mean(),2)
    Main_n2_price = round(df[df['neighbourhood']== Neigbourhood2]['price'].dropna().mean(),2)
    cheap_Or_Expensive = 'Cheap' if Main_n1_price < Main_n2_price else 'Expensive'
    percent = round(abs(((Main_n2_price*100)/Main_n1_price)-100),2)
        
    if Main_n1_price == Main_n2_price:
        txt = ['Both neigbourhood have he same average price', r'0% of diference']
    else:
        txt = [f'The {Neigbourhood1} district has on avarage {Main_n1_price}, and the district of {Neigbourhood2} has on avarage {Main_n2_price}', f' It means the neigborhood {Neigbourhood1} is {cheap_Or_Expensive} more in {percent}% in relation of {Neigbourhood2}']
    return txt


def MappingPlot(df, mimimun, maximun):
    #Variable to map
    df_map = df
    #Color Scale to help to identify some important variables
    color_scale = [(0, 'green'), (1,'red')]
    #New Dataframe Considering Outliers Treatment
    df_outliers_removed = df_map[(df_map["price"] < maximun) & (df_map["price"] > mimimun)]
    #Figure
    fig = px.scatter_mapbox(df_outliers_removed, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_name="neighbourhood", 
                            hover_data=["neighbourhood", "price"],
                            color="price",
                            color_continuous_scale=color_scale,
                            zoom = 10,
                            height=300,
                            width=1500)

    #Map Style
    fig.update_layout(mapbox_style="open-street-map")
    #Margins settings
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #Figure show
    return fig

def MappingHistogramPlot(df, mimimun, maximun):
    #Variable to map
    df_map = df
    #Color Scale to help to identify some important variables
    color_scale = [(0, 'green'), (1,'red')]
    #New Dataframe Considering Outliers Treatment
    df_outliers_removed = df_map[(df_map["price"] < maximun) & (df_map["price"] > mimimun)]
    #Figure
    fig = px.histogram(df_outliers_removed, 
                        x='price', 
                        width=1500, 
                        height=300)
    return fig
def Model_Adjust(df, input):
    X, y = df.drop('price', axis= 1), df['price']
    X_Train, X_Test, Y_Train, Y_Test = train_test_split (X, y, test_size=0.2, random_state=0)
    model = ExtraTreesRegressor(
    bootstrap=False,
    ccp_alpha=0.0, 
    criterion='friedman_mse',
    max_depth=None, 
    max_features='auto', 
    max_leaf_nodes=None,
    max_samples=None, 
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=2, 
    min_weight_fraction_leaf=0.0, 
    n_jobs=-1, 
    oob_score=False, 
    verbose=0, 
    warm_start=False, 
    n_estimators=100,
    random_state=3840).fit(X_Train, Y_Train)
    return model.predict(input)[0], model.score(X_Test, Y_Test)
    
st.markdown('# Streamlit Solution')
tab1, tab2, tab3 = st.tabs(["Mapping Solution", "Calc. Price Prediction", "Neighbourhood Comparation"])

with tab1:
   st.header("Mapping of expensive prices")
   Percentil = st.slider("Select a range of values based in percentil", 0.0, 100.0, (0.0, 95.0))
   q_low = df["price"].quantile((Percentil[0]/100))
   q_hi  = df["price"].quantile((Percentil[1]/100))
   st.plotly_chart(MappingPlot(df, q_low, q_hi))
   st.plotly_chart(MappingHistogramPlot(df, q_low, q_hi))

with tab2:
   st.header("Price Prediction")
   with st.form("Calc_form"):

    st.write("Information That We need:")
    col1, col2 = st.columns(2)
    with col1:
        minimum_nights = st.number_input("Minimun Nights*", label_visibility = "visible", step  = 1)
        number_of_reviews = st.number_input("Number Review*", label_visibility = "visible", step  = 1)
        reviews_per_month = st.number_input("Review in a Month*", label_visibility = "visible", step  = 1)
        calculated_host_listings_count = st.number_input("Host in list*", label_visibility = "visible", step  = 1)
    with col2:
        availability_365 = st.number_input("Avaliablity in one year*", label_visibility = "visible", step  = 1)
        number_of_reviews_ltm = st.number_input("review in one year*", label_visibility = "visible", step  = 1)
        there_is_licence = st.checkbox("Licence")
        diff_days = st.number_input("Days from last review*", label_visibility = "visible", step  = 1)
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Calculation")
    if submitted:
        input=[[
                float(minimum_nights if minimum_nights != 0 else 0),
                float(number_of_reviews if number_of_reviews!= 0 else 0),
                float(reviews_per_month if reviews_per_month!= 0 else 0),
                float(calculated_host_listings_count if calculated_host_listings_count!= 0 else 0),
                float(availability_365 if availability_365!= 0 else 0),
                float(number_of_reviews_ltm if number_of_reviews_ltm!= 0 else 0),
                float(0 if there_is_licence == False else 1),
                float(diff_days if diff_days!= 0 else 0)
             ]]
        price_Predicted, model_Adjust_r2 = Model_Adjust(df_model_base, input)
        st.write(f'The price Predict was: ${round(price_Predicted,2)}')
        st.write(f'The Model R² adjust was: {round(model_Adjust_r2,4)*100}%')

with tab3:
    st.header("Price Comparation")
    st.write("Select 2 Neighbourhood")
    col1, col2 = st.columns(2)
    with col1:
        Ne1 = st.selectbox('Choose a Neighbourhood 1', options= Neighbourhood_choices)  
    with col2:
        Ne2 = st.selectbox('Choose a Neighbourhood 2', options= Neighbourhood_choices)
    if st.button('Compare'):
        try:
            txt = Compare(df,Ne1, Ne2)
            st.write(txt[0])
            st.write(txt[1])
        except:
            txt = ['ERROR: Both neighbourhood have the same name', 'No calculations based in the Error']
            st.write(txt[0])
            st.write(txt[1])
    else:
        st.write('Not Select yet')
    

#tab1, tab2 = st.tabs(["Cat", "Dog"])

#with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    #
    
#with tab2:
    # Use the native Plotly theme.
    #st.plotly_chart(px.histogram(df[(df["price"] < q_hi) & (df["price"] > q_low)], x='price'))

