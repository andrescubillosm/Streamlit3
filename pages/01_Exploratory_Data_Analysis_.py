##################################################
## {Description: Capstone project for DS4A Colombia and Eficacia S.A.}
##################################################
## {License_info: Restricted For educational proporses}
##################################################
## Author: {Team 108 DS4A Colombia}
## Copyright: Copyright {2022}, {Efficacia DS APP}
## Credits: [{credit_list: Team 108 :Andres Felipe Garcia, Camilo Gomez, Carlos Andres Cubillos , Ilan Almaza, Luis Miguel Puerto, Ricardo Leon, Ricardo Rodriguez, ; Data from eficacia S.A.; soport Correlation one DS4A colombia}]
## License: {license}
## Version: {mayor}.{minor}.{rel}
## Maintainer: {Unmaintained}
## Email: {contact_email}
## Status: {dev_status: under development}
##################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title(":bar_chart: EXPLORATORY DATA ANALYSIS")
st.markdown("For the exploratory data analysis, different data cleaning, filtering and grouping techniques were applied to the datasets received, it was necessary to discard some datasets and some columns in order to reduce their size and make them easier to process, then you can interact with some of the main findings by means of a series of filters and selections.")

#sidebar
st.sidebar.markdown("Developed by team 108 :globe_with_meridians: for DS4A Colombia cohort 6.")
st.sidebar.write(f'''
    <a target="_blank" href="https://main.d1apizaihpmcr4.amplifyapp.com/">
        <button>
            Return to project page
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)
st.sidebar.markdown(" &copy; 2022 &copy;")


#creating horizontal containers
submenus= st.container()
Expo_1= st.container()
Expo_2= st.container()
Expo_3= st.container()
Expo_4= st.container()


#Adding datasets
#url_1 = 'https://eficaciadata.s3.amazonaws.com/geodata.csv' # Data of geographic points of the stores
#url_2 = 'https://eficaciadata.s3.amazonaws.com/pro_pre_pdv.csv' # Data of products-Prices-stores
#url_3 = 'https://eficaciadata.s3.amazonaws.com/pre.csv'# Data of prices
#url_4 = 'https://eficaciadata.s3.amazonaws.com/pro.csv' # DATA DE PRODUCTOS
#url_5 = 'https://eficaciadata.s3.amazonaws.com/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#url_6 = 'https://eficaciadata.s3.amazonaws.com/ago.csv' # DATA DE AGOTADOS

#url_1 = 'csv/geodata.csv' # Data of geographic points of the stores
#url_2 = 'csv/pro_pre_pdv.csv' # Data of products-Prices-stores
#url_3 = 'csv/pre.csv'# Data of prices
#url_4 = 'csv/pro.csv' # DATA DE PRODUCTOS
#url_5 = 'csv/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#url_6 = 'csv/ago.csv' # DATA DE AGOTADOS
#uses this instruccion it the data change @st.cache(persist=True)( If you have a different use case where the data does not change so very often, you can simply use this)

#loading data
#gea = pd.read_csv(url_1, sep=';')
#data_2 = pd.read_csv(url_2)
#data_3 = pd.read_csv(url_3)
#data_4 = pd.read_csv(url_4)
#data_5 = pd.read_csv(url_5)
#data_6 = pd.read_csv(url_6)

#parquets
url_1 = 'parquet/geodata.parquet' # Data of geographic points of the stores
url_2 = 'parquet/pro_pre_pdv.parquet' # Data of products-Prices-stores
url_3 = 'parquet/pre.parquet'# Data of prices
url_4 = 'parquet/pro.parquet' # DATA DE PRODUCTOS
url_5 = 'parquet/ago_pdv_pro.parquet' # Data of products out stock for store and type of product
url_6 = 'parquet/ago.parquet' # DATA DE AGOTADOS

#uses this instruccion it the data change @st.cache(persist=True)( If you have a different use case where the data does not change so very often, you can simply use this)


#loading data
gea = pd.read_parquet(url_1, engine='auto')
data_2 = pd.read_parquet(url_2, engine='auto')
data_3 = pd.read_parquet(url_3, engine='auto')
data_4 = pd.read_parquet(url_4, engine='auto')
data_5 = pd.read_parquet(url_5, engine='auto')
data_6 = pd.read_parquet(url_6, engine='auto')

ANTIOQUIA = 'Medellin','Caucasia','El Bagre','Zaragoza','La Ceja','Rionegro','Bello','Copacabana','Marinilla','Itagui','Envigado','Caldas','Sabaneta','Turbo','Apartadó','Carepa','La Estrella','San Jerónimo','Barbosa','Fredonia','Amagá','Necoclí','La Unión','Carmen De Viboral','Retiro'
#ARAUCA = "Arauca"
ATLANTICO = 'Barranquilla','Soledad','Malambo','Sabanalarga','Candelaria','Galapa','Puerto Colombia','Santo Tomás','Sabanagrande','Palmar De Varela','Baranoa'
#BOGOTADC = "Bogotá"
BOLIVAR = 'Cartagena','Magangue','Arjona','Turbaco','San Juan Nepomuceno','Mompos','Carmen De Bolívar'
BOYACA ='Tunja','Duitama','Sogamoso','Paipa','Chiquinquirá','Villa De Leyva','Samacá','Puerto Boyacá'
CALDAS = 'Manizales','La Dorada','Villamaría','Chinchiná'
#CAQUETA= 'Florencia'
CASANARE = 'Yopal'
CAUCA = 'Popayán','Santander De Quilichao','Puerto Tejada'
CESAR = 'Valledupar','Aguachica','Bosconia','Curumaní'
CHOCO= 'Quibdó'
CORDOBA = 'Montería','Cereté','La Apartada','Lorica','Planeta Rica','Sahagún','Ciénaga De Oro','Tierralta'
CUNDINAMARCA= 'Chía','Girardot','Facatativá','Mosquera','Soacha','Zipaquirá','Madrid','Cajicá','Fusagasugá','La Calera','Ricaurte','Cota','La Mesa','Villeta','Villa De San Diego De Ubate','Chocontá','Funza','Gachancipá','Pacho','Tocancipá','Villa Pinzón','Tocaima'
HUILA = 'Neiva','Pitalito','San Agustín','Campoalegre'
LAGUAJIRA = 'Riohacha','Albania','Fonseca','Maicao','San Juan Del Cesar'
MAGDALENA= 'Santa Marta','Aracataca','Pivijay','El Banco','Ciénaga','Fundación'
META = 'Villavicencio','Restrepo','Acacías'
NARINO = 'Pasto','Ipiales'
NORTEDESANTANDER = 'Cúcuta','Ocaña'
QUINDIO = 'Armenia','Calarca','Montenegro','Quimbaya','La Tebaida','Circasia'
RISARALDA = 'Pereira','Dosquebradas','Santa Rosa De Cabal'
SANTANDER = 'Floridablanca','Bucaramanga','Barrancabermeja','Piedecuesta','San Gil','Cerrito','San Miguel'
SUCRE ='Sincelejo','Corozal','Toluviejo','Coveñas','Sampués','San Onofre'
TOLIMA ='Flandes','Ibagué','Espinal','Melgar'
VALLEDELCAUCA ='Santiago De Cali','Palmira','Jamundí','Guadalajara De Buga','Cartago','Tulúa','Buenaventura','Guacarí','Caicedonia','Zarzal','Yumbo','Bugalagrande','Ginebra','Roldanillo','Sevilla'



#submenus
with submenus:
     st.title("Select options:")
     select_depto = st.selectbox('Select a department of Colombia',
                                ('ANTIOQUIA','ARAUCA','ATLÁNTICO',
                                 'BOGOTA D.C','BOLIVAR','BOYACA',
                                 'CALDAS','CAQUETA','CASANARE',
                                 'CAUCA','CESAR','CHOCO','CORDOBA',
                                 'CUNDINAMARCA','HUILA','LA GUAJIRA',
                                 'MAGDALENA','META','NARIÑO','NORTE DE SANTANDER',
                                 'QUINDÍO','RISARALDA','SANTANDER','SUCRE','TOLIMA','VALLE DEL CAUCA'))
     if select_depto == 'ANTIOQUIA' :
        select_ciudad = st.selectbox(
        'Select a city',(ANTIOQUIA))
     elif select_depto == 'ARAUCA' :
         select_ciudad = 'Arauca'
     elif select_depto == 'ATLÁNTICO':
         select_ciudad = st.selectbox(
         'Select a city',(ATLANTICO))
     elif select_depto == 'BOGOTA D.C' :
         select_ciudad="Bogotá"
     elif select_depto == 'BOLIVAR' :
         select_ciudad = st.selectbox(
         'Select a city',(BOLIVAR))
     elif select_depto == 'BOYACA' :
         select_ciudad = st.selectbox(
         'Select a city',(BOYACA))
     elif select_depto == 'CALDAS' :
         select_ciudad = st.selectbox(
         'Select a city',(CALDAS))
     elif select_depto == 'CAQUETA' :
         select_ciudad="Florencia"
     elif select_depto == 'CASANARE' :
         select_ciudad = st.selectbox(
         'Select a city',"Yopal")
     elif select_depto == 'CAUCA' :
         select_ciudad = st.selectbox(
         'Select a city',(CAUCA))
     elif select_depto == 'CESAR' :
         select_ciudad = st.selectbox(
         'Select a city',(CESAR))
     elif select_depto == 'CHOCO' :
         select_ciudad = st.selectbox(
         'Select a city',(CHOCO))
     elif select_depto == 'CORDOBA' :
         select_ciudad = st.selectbox(
         'Select a city',(CORDOBA))
     elif select_depto == 'CUNDINAMARCA' :
         select_ciudad = st.selectbox(
         'Select a city',(CUNDINAMARCA))
     elif select_depto == 'LA GUAJIRA' :
         select_ciudad = st.selectbox(
         'Select a city',(LAGUAJIRA))
     elif select_depto == 'MAGDALENA' :
         select_ciudad = st.selectbox(
         'Select a city',(MAGDALENA))
     elif select_depto == 'META' :
         select_ciudad = st.selectbox(
         'Select a city',(META))
     elif select_depto == 'NARIÑO' :
         select_ciudad = st.selectbox(
         'Select a city',(NARINO))
     elif select_depto == 'NORTE DE SANTANDER' :
         select_ciudad = st.selectbox(
         'Select a city',(NORTEDESANTANDER))
     elif select_depto == 'QUINDÍO' :
         select_ciudad = st.selectbox(
         'Select a city',(QUINDIO))
     elif select_depto == 'RISARALDA' :
         select_ciudad = st.selectbox(
         'Select a city',(RISARALDA))
     elif select_depto == 'SANTANDER':
         select_ciudad = st.selectbox(
         'Select a city',(SANTANDER))
     elif select_depto == 'SUCRE' :
         select_ciudad = st.selectbox(
         'Select a city',(SUCRE))
     elif select_depto == 'TOLIMA' :
         select_ciudad = st.selectbox(
         'Select a city',(TOLIMA))
     elif select_depto == 'VALLE DEL CAUCA' :
         select_ciudad = st.selectbox(
         'Select a city',(VALLEDELCAUCA))
     data_5_fil= data_5.loc[data_5['Ciudad'] == select_ciudad]
     select_cadena = st.selectbox('Select a brand of stores for this city', options=pd.unique(data_5_fil['Cadena']))
     data_5_cad = data_5_fil.loc[data_5_fil['Cadena'] == select_cadena]
     select_categoria = st.selectbox('Select a product category',	options=pd.unique(data_5_cad['Categoria']))
     data_5_cat = data_5_cad.loc[data_5_cad['Categoria'] == select_categoria]
     select_subcategoria = st.selectbox('Select a sub-category',	options=pd.unique(data_5_cat['SubCategoria']))
     data_5_subcat = data_5_cat.loc[data_5_cat['SubCategoria'] == select_subcategoria]



with Expo_1:
    st.title("Exploratory graphs")
    # Grafica del número de eventos de productos agotados registrados por fecha y ciudad (solo las 10 ciudades con más agotados)
    st.subheader("Charts organized by city")
    st.markdown("Top cities outstocks")
    data_5['Ciudad'] = data_5.Ciudad.astype(str)
    data_5['Cadena'] = data_5.Cadena.astype(str)
    top_ciudades = data_5.groupby("Ciudad")['Agotado'].sum().sort_values(ascending= False).head(10).index
    agotados_top_ciudades = data_5.loc[data_5['Ciudad'].isin(top_ciudades)]
    fig, ax = plt.subplots(figsize = (15,10))
    asd = data_5.loc[data_5['Ciudad'].isin(top_ciudades)].pivot_table(
    index = 'Fecha',
    columns = 'Ciudad',
    values = 'Agotado',
    aggfunc = np.sum
    ).plot(kind = 'line', ax = ax)
    ax.set(
    ylabel = 'number',
    title = 'Registered number of stockouts by city and date'
    )
    st.pyplot(fig)


with Expo_2:
    # Grafica del número de eventos de productos agotados registrados por fecha y Cadena (solo las 10 Cadenas con más agotados)
    st.markdown("___")
    st.subheader("Charts organized by type of store chain or point of sale")
    st.markdown("Top brands or sales chains with outstocks")
    top_cadenas = data_5.groupby("Cadena")['Agotado'].sum().sort_values(ascending= False).head(10).index
    agotados_top_cadenas = data_5.loc[data_5['Cadena'].isin(top_cadenas)]
    fig, ax = plt.subplots(figsize = (15,10))
    agotados_cadenas = agotados_top_cadenas.loc[data_5['Cadena'].isin(top_cadenas)].pivot_table(
    index = 'Fecha',
    columns = 'Cadena',
    values = 'Agotado',
    aggfunc = np.sum
    ).plot(kind = 'line', ax = ax)
    ax.set(
    ylabel = 'number',
    title = 'Registered number of stockouts by retailer and date'
    )
    st.pyplot(fig)
    st.markdown("Top brands or sales chains with outstocks by city: "+ select_ciudad.title())
    data_5_fil['Cadena'] = data_5_fil.Cadena.astype(str)
    top_cadenas = data_5_fil.groupby("Cadena")['Agotado'].sum().sort_values(ascending= False).head(10).index
    agotados_top_cadenas = data_5_fil.loc[data_5_fil['Cadena'].isin(top_cadenas)]
    fig, ax = plt.subplots(figsize = (15,10))
    agotados_cadenas = agotados_top_cadenas.loc[data_5_fil['Cadena'].isin(top_cadenas)].pivot_table(
    index = 'Fecha',
    columns = 'Cadena',
    values = 'Agotado',
    aggfunc = np.sum
    ).plot(kind = 'line', ax = ax)
    ax.set(
    ylabel = 'number',
    title = 'Registered number of stockouts by retailer and date'
    )
    st.pyplot(fig)

with Expo_3:
    st.markdown("___")
    st.subheader("Charts organized by outstock")
    st.markdown("Causes of product stockouts all country(dont chage with any filter)")
    fig, ax = plt.subplots(figsize = (18,10))
    agotados = data_5.loc[data_5['Agotado'] == True]
    agotados.pivot_table(
    index = 'Fecha',
    columns ='SubCausal',
    values = 'Agotado',
    aggfunc = np.sum

    ).plot(kind = 'line', ax = ax)
    st.pyplot(fig)
    st.markdown("Causes of product stockouts by city: " + select_ciudad.title())
    fig, ax = plt.subplots(figsize = (18,10))
    agotados_ciudad = data_5_fil.loc[data_5_fil['Agotado'] == True]
    agotados_ciudad.pivot_table(
    index = 'Fecha',
    columns ='SubCausal',
    values = 'Agotado',
    aggfunc = np.sum
    ).plot(kind = 'line', ax = ax)
    st.pyplot(fig)

    st.markdown("Causes of product stockouts by Category: " +select_categoria.title())
    fig, ax = plt.subplots(figsize = (18,10))
    agotados_cat = data_5_cat.loc[data_5_cat['Agotado'] == True]
    agotados_cat.pivot_table(
    index = 'Fecha',
    columns ='SubCausal',
    values = 'Agotado',
    aggfunc = np.sum

    ).plot(kind = 'line', ax = ax)
    st.pyplot(fig)
    st.markdown("Causes of product stockouts by subcategories of products: "+ select_subcategoria.title())
    fig, ax = plt.subplots(figsize = (18,10))
    agotados_subcat = data_5_subcat.loc[data_5_subcat['Agotado'] == True]
    agotados_subcat.pivot_table(
    index = 'Fecha',
    columns ='SubCausal',
    values = 'Agotado',
    aggfunc = np.sum

    ).plot(kind = 'line', ax = ax)
    st.pyplot(fig)




