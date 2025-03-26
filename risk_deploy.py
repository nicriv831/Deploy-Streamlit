from src import *

from sklearn import set_config
set_config(transform_output='pandas')

# INTRO
st.title('Credit Default Risk Predictor')
st.write("Let's see how likely a client is to have payment problems")

# load dataframe

@st.cache_data
def load_data():
  return pd.read_csv('Data/Raw_Data/Filtered_Train_App.csv')

trained = load_data()

# Load model and pipeline

@st.cache_resource
def load_model():
  return joblib.load('1credit_default_model.joblib')

model = load_model()

def load_pipe():
  return joblib.load('1credit_default_pipeline.joblib')

main_pipe = load_pipe()

prediction = pd.DataFrame(columns=['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
       'ORGANIZATION_TYPE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_ANNUITY', 'EXT_SOURCE_2', 'CREDIT_ACTIVE', 'CREDIT_CURRENCY',
       'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG',
       'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE', 'DAYS_CREDIT_UPDATE',
       'MOST_RECENT', 'DAYS_CREDIT_REC', 'MONTHS_BALANCE', 'STATUS',
       'PREV_NAME_CONTRACT_TYPE', 'AMT_APPLICATION',
       'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
       'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION',
       'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE',
       'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
       'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY',
       'NAME_YIELD_GROUP', 'PREV_APP_COUNT', 'PREV_APP_BINS'])

for feat in trained.columns:
   if feat != 'TARGET':
     if feat == 'DAYS_EMPLOYED':
       prediction.loc[0, feat] = (st.number_input(label='**How many days have they been employed?** _(use +/- or type in, will convert to negative number at bottom)_', min_value=0, max_value = 100000, step = 1 ) * -1)
     elif feat == 'FLAG_EMP_PHONE':
       answer = st.selectbox(label='**Do they have a work phone number?** _(select one)_', options=('Yes', 'No'))
       if answer == 'Yes' : prediction.loc[0, feat] = 1
       elif answer == 'No' : prediction.loc[0, feat] = 0
     elif feat == 'NAME_INCOME_TYPE':
       prediction.loc[0, feat] = st.selectbox(label = '**What type of income do they have?** _(select one)_',
                                              options=('Unemployed', 'Working', 'Commercial associate', 'Pensioner', 'State servant'))
     elif feat == 'ORGANIZATION_TYPE':
       prediction.loc[0, feat] = st.selectbox(label = '**What type of organization do they work for?** _(select one)_',
                                              options=('Business Entity Type 1', 'Business Entity Type 2', 'Business Entity Type 3',
                                                       'Transport: type 1', 'Transport: type 2', 'Transport: type 3', 'Transport: type 4',
                                                       'Trade: type 1', 'Trade: type 2','Trade: type 3', 'Trade: type 4','Trade: type 5','Trade: type 6','Trade: type 7',
                                                       'Industry: type 1', 'Industry: type 2','Industry: type 3', 'Industry: type 4','Industry: type 5','Industry: type 6',
                                                       'Industry: type 7', 'Industry: type 8','Industry: type 9','Industry: type 10', 'Industry: type 11', 'Industry: type 12','Industry: type 13',
                                                       'XNA', 'Self-employed', 'Other', 'Medicine', 'Government', 'School', 'Kindergarten', 'Construction', 'Security', 'Housing', 'Military',
                                                       'Bank', 'Agriculture', 'Police', 'Postal', 'Security', 'Restaurant', 'Services', 'University', 'Hotel', 'Electricity', 'Insurance',
                                                       'Telecom', 'Emergency', 'Advertising', 'Realtor', 'Culture', 'Mobile', 'Legal', 'Cleaning','Religion'))
    #  elif feat == 'NAME_GOODS_CATEGORY':
    #    prediction.loc[0, feat] = #

st.dataframe(prediction)


if st.button('Predict'):
  prediction = main_pipe.transform(prediction)
  default = (model.predict_proba(prediction)[:,1]) * 100

  if default > 50:
    st.markdown(f'## They have a {default}% chance of having payment issues - Default risk is {default}%')
  else:
    st.markdown(f'## They have a less than 50% chance of having a payment problem - Default risk is {default}%')
