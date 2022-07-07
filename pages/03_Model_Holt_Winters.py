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
from statsmodels.tsa.holtwinters import Holt
import plotly.graph_objects as go

st.title(" :paperclip: Model of time series Holt-Winters")
st.markdown("A model of time series behavior is Holt-Winters. Holt-Winters is a way to model three characteristics of the time series: a typical value (average), a slope (trend over time), and a cyclical repeating pattern. Forecasting always requires a model (seasonality).")
st.markdown("Time series anomaly detection is a challenging issue with many workable solutions. With all of the subjects it covers, it's simple to become confused. Although it can be difficult to learn them, it is frequently more difficult to put them into practice. A crucial component of anomaly detection is forecasting, which is using information about a time series—either from a model or its past—to make predictions about future values. There is another choice. With exponential smoothing, you can accomplish something that is much easier. ")
st.markdown("for more information on this method, see this [page](https://orangematter.solarwinds.com/2019/12/15/holt-winters-forecasting-simplified/) or consult this page of [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.Holt.html?highlight=holt#statsmodels.tsa.holtwinters.Holt) which has several options on this model or method ")

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

submenus= st.container()
model = st.container()
predictions_mod= st.container()


#Adding datasets
url_7 =  pd.read_parquet('parquet/ago_pro_pdv.parquet', engine='auto')
#url_5 = 'https://eficaciadata.s3.amazonaws.com/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#url_5 = 'csv/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#loading data
#data_7 = pd.read_csv(url_5)

#SUBMENUS
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


#formula used
with submenus:
     st.title("Select options for the model")
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

     data_7_fil= url_7.loc[url_7['ciudad'] == select_ciudad]

     select_cadena = st.selectbox('Select a brand of stores for this city', options=pd.unique(data_7_fil['cadena']))
     data_7_cad = data_7_fil.loc[data_7_fil['cadena'] == select_cadena]
     select_categoria = st.selectbox('Select a product category',	options=pd.unique(data_7_cad['categoria']))
     data_7_cat = data_7_cad.loc[data_7_cad['categoria'] == select_categoria]
     #select_subcategoria = st.selectbox('Select a sub-category',	options=pd.unique(data_7_cat['SubCategoria']))
     #data_7_subcat = data_7_cat.loc[data_7_cat['SubCategoria'] == select_subcategoria]

def ts_factory(agotados, cadena = 'Todas', ciudad = 'Todas', categoria = 'Todas'):

    if (cadena == 'Todas') & (ciudad == 'Todas') & (categoria == 'Todas'):

        ts = agotados.groupby(agotados['fecha'].dt.to_period('W'))['agotado'].sum()
        ts.index = ts.index.to_timestamp()

    else:
        queries = []
        if cadena != 'Todas':
            queries.append('(cadena == @cadena)')
        if ciudad != 'Todas':
            queries.append('(ciudad == @ciudad)')
        if categoria != 'Todas':
            queries.append('(categoria == @categoria)')

        final_query = '&'.join(queries)
        df_agotados = agotados.query(final_query)
        ts = df_agotados.groupby(df_agotados['fecha'].dt.to_period('W'))['agotado'].sum()
        ts.index = ts.index.to_timestamp()

    return ts

def ts_plotter(agotados, cadena = 'Todas', ciudad = 'Todas', categoria = 'Todas'):

    ts = ts_factory(agotados, cadena, ciudad, categoria)

    fitted_model = Holt(ts, initialization_method = 'estimated').fit(smoothing_level = 0.8)
    model_pre = np.abs(pd.concat([fitted_model.fittedvalues, fitted_model.forecast(4)]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x = ts.index, y = ts, line = dict(dash = 'solid', color = 'black'), name = 'Datos reales')
    )
    fig.add_trace(
        go.Scatter(x = model_pre.index, y = model_pre, line = dict(dash = 'dash', color = 'red'), name = 'Modelo')
    )
    fig.update_layout(title = f'Agotados semanales en ciudad : {ciudad}, cadena : {cadena}, categoria: {categoria} ',
                     xaxis_title = 'Fecha',hovermode = 'x unified')
    fig.update_xaxes(dict(type = 'date', range = ('2021-09-27 00:00:00', '2022-06-15 00:00:00')))

    return fig

#graficos
st.write(ts_plotter(url_7, ciudad = select_ciudad))
st.write(ts_plotter(url_7, ciudad = select_ciudad, cadena = select_cadena))
st.write(ts_plotter(url_7, ciudad = select_ciudad, cadena = select_cadena, categoria = select_categoria))
