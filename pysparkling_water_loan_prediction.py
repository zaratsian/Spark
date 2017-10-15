
###############################################################################################################
#
#   H2O Sparkling Water
#   https://www.h2o.ai/download/
#   http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#hadoop-users
#   http://docs.h2o.ai/h2o/latest-stable/index.html
#
#   Simple example to predict loan defaults (note: minimal data prep was done for this example)
#   Lending Club Loan Dataset: https://www.kaggle.com/wendykan/lending-club-loan-data
#
###############################################################################################################

from pysparkling import *
hc = H2OContext.getOrCreate(spark)
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from pyspark.sql.types import *
from pyspark.sql.functions import *


# Import Data
# https://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/docs-website/h2o-py/docs/h2o.html
loans = h2o.import_file(            \
        path="/loan.csv"            \
        ,header=0                   \
        #,sep=',',                  \
        #,col_names=column_names,   \
        #,col_types=column_types    \
        )

# Data Descriptive Info
loans.columns
loans.head()
loans.types
loans.summary()
loans.shape

# Drop unneeded columns and filter NA columns
# Define schema, this is used to keep variables. 
# It would also be used to ingest data to a specific format (when possible and depending on number of columns)
keep_schema =  {
                    "id":"string"
                    ,"member_id":"string"
                    ,"loan_amnt":"float"
                    ,"term":"int"
                    ,"int_rate":"float"
                    ,"installment":"string"
                    ,"grade":"string"
                    #,"sub_grade":"string"
                    ,"emp_length":"int"
                    ,"home_ownership":"int"
                    ,"annual_inc":"float"
                    ,"loan_status":"int"
                    #,"purpose":"string"
                    #,"fico_range_high":"int"
                    #,"fico_range_low":"int"
                    #,"zip_code":"string"
                    ,"addr_state":"string"
                    #,"delinq_2yrs":"int"
                    #,"open_acc":"int"
                    ,"total_acc":"int"
                    ,"open_acc":"int"
                    ,"tot_coll_amt":"float"
                    #,"revol_util":"float"    # Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
                }

keep_columns = [col[0] for col in keep_schema.iteritems()]

loans.shape
loans = loans.drop([col for col in loans.columns if col not in keep_columns])
loans.shape
columns_to_keep = loans.filter_na_cols(frac=0.2)  # Returns a list of indices of columns that have fewer NAs than "frac". If all columns are filtered, None is returned.
loans = loans.drop([col for i,col in enumerate(loans.columns) if i not in columns_to_keep]) if [col for i,col in enumerate(loans.columns) if i not in columns_to_keep] != [] else loans
loans.shape

loans["addr_state"].table()
loans["loan_status"].table()

# Restructure Target Variable (i.e. loan default or not)

# Convert from H2O DF to Spark DF
df_loans = hc.as_spark_frame(loans,)
# Convert from Spark DF to H20 DF
#loans = hc.as_h2o_frame(df_loans)
df_loans = df_loans.withColumn("loan_status", when((df_loans["loan_status"]=="Default") | (df_loans["loan_status"]=="Charged Off"), "default").when((df_loans["loan_status"]=="Fully Paid") | (df_loans["loan_status"]=="Current"), "no default").otherwise(df_loans["loan_status"]) )
df_loans.count()
df_loans = df_loans.filter( (df_loans["loan_status"] == "default") | (df_loans["loan_status"] == "no default") )
df_loans.count()

# Create equal split of "default" and "no default" records
number_of_no_defaults = df_loans.groupBy("loan_status").count().filter(df_loans["loan_status"]=="no default").collect()[0][1]
number_of_defaults    = df_loans.groupBy("loan_status").count().filter(df_loans["loan_status"]=="default").collect()[0][1]
stratified_ratio      = number_of_defaults / float(number_of_no_defaults)
df_loans.groupBy("loan_status").count().show()
df_loans = df_loans.sampleBy("loan_status", fractions={"default": 1.00, "no default": stratified_ratio}, seed=0)
df_loans.groupBy("loan_status").count().show()

# Convert from Spark DF to H20 DF
loans = hc.as_h2o_frame(df_loans)

# H2O Change Data type
# Change "string" to "enum" for H2O dataframe
string_columns = [col[0] for col in loans.types.iteritems() if col[1]=='string']
for col in string_columns:
    loans[col] = loans[col].asfactor()  # Convert to factor / enum

loans.summary()

# Loan Status Distributions
loans["loan_status"].table()

# Split H2O data table into train test and ** validation ** datasets
training, testing, validation = loans.split_frame([0.60,0.20],seed=12345)
training.shape
testing.shape
validation.shape


# Specify Target
target = "loan_status"

# Specify Predictors
predictors = loans.columns[:]
predictors.remove(target)
predictors.remove("id")
predictors.remove("member_id")

# Modeling
model_gbm = H2OGradientBoostingEstimator(ntrees=50, max_depth=6, learn_rate=0.1, distribution="bernoulli")
model_gbm.train(x=predictors, y=target, training_frame=training)
model_gbm.varimp(True)
model_gbm.confusion_matrix(train = True)
model_gbm.auc(train=True)
model_gbm.model_performance(testing)
predictions = model_gbm.predict(testing)
predictions.head()

# Simple Deep Learning - Predict Arrest
model_dl = H2ODeepLearningEstimator(variable_importances=True, loss="Automatic")
model_dl.train(x=predictors, y=target, training_frame=training, validation_frame=validation)
model_dl.varimp(True)
model_dl.confusion_matrix(train = True)
model_dl.auc(train=True)
model_dl.model_performance(testing)
predictions = model_dl.predict(testing)
predictions.head()

# Save Model
model_path = h2o.save_model(model=model_gbm, path="/tmp/", force=True)
print model_path




# Python web framework setup:

id                  = 12345
member_id           = 12345
loan_status         = 'no default'
loan_amnt           = 10000
term                = "60 months"
int_rate            = 12.5
installment         = 300
grade               = "C"
emp_length          = "5 years"
home_ownership      = "RENT"
annual_inc          = 40000
addr_state          = "CA"
total_acc           = 15
tot_coll_amt        = 10000

input_column_names = [ "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "addr_state", "total_acc", "tot_coll_amt" ]

test_input = h2o.H2OFrame.from_python( [(loan_amnt, term, int_rate, installment, grade, emp_length, home_ownership, annual_inc, addr_state, total_acc, tot_coll_amt)], column_names=input_column_names)
test_input.head()

# Load the model
# model_gbm = h2o.load_model(model_path)
result = model_gbm.predict(test_input)
default_probability = result.as_data_frame()['default'][0]


#ZEND
