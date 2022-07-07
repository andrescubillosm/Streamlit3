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

st.title(":blue_book: Conclusions and  Closing Remarks")
st.markdown("In this section you will find a summary or the main findings of the project, it should be noted that this project is for educational purposes only, if you have questions about the data you should ask directly to the owner of these that is the company Eficacia.SA, the members of the team 108 will be willing to answer questions about the conclusions of the project see project page to contact them.")

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
data_exp=st.container()
why_mod= st.container()
error_ex= st.container()
conclud_final= st.container()



#conclusions about data
with data_exp:
    st.title("Data")
    st.markdown("We are grateful in advance for the efforts made by the company Eficacia and its willingness to assist in the process. ")
    st.markdown("In conclusion, the data show problems with some stores in the collection or updating of information, particularly the small chains, which report products out of stock for longer periods of time than the other chains; the inventory data are necessary for analysis or the possible application of more complex models, but in this case they were discarded because it was not possible to relate them for various reasons.")
    st.markdown("There is a general imbalance in the data, but as mentioned above, it seems to be caused by the point at which the information is collected or reports inventories, prices and others, or by the fact that when data is processed (filtering, cleaning, organizing and joining), there are errors by specific city, by category and by chain, so it is suggested that processes be standardized and automated to eliminate this imbalance and make it easier to implement data management mechanisms.")



#why we used that model
with why_mod:
    st.title("Other model options")
    st.markdown("Although several types of models were analyzed for better forecasting, the quality of the data should be improved, starting with the collection process itself, so that in the future it will be easier to apply other model options in order to be more effective in forecasting possible inventory stockouts.  It would also be important to add an inventory dataset for each brand or chain of warehouses so that the information could be trained or contrasted with more complex models or algorithms.")
    st.markdown("Several models were tested using the 'AutoTS' library, from which the theta model was chosen as the best performing one, but even so, the model does not give a satisfactory result given the validity of the data, which has some inconsistencies, as explained in other sections, Therefore, the Holt-Winters smoothing model using the statsmodels library gives a result more in line with the desired result where it predicts which chains and in which cities products are out of stock with a variation, but this model requires an adjustment from the data and where possible a training with a larger portion of data and that these are balanced and as has been recommended since the collection should be standardized for a good result.")


#conclusions
with error_ex:
    st.title("Conclusions")
    st.markdown("Conclusions")
    st.markdown("As an additional recommendation, a robust local or cloud-based production or deployment environment should be considered, which may generate costs depending on the size of the data and application deployed and the type of service purchased, given that the performance of the analysis, model and deployment requires memory and processing resources.")