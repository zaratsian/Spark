
###############################################################################################################
#
#   H2O Sparkling Water
#   https://www.h2o.ai/download/
#   http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#hadoop-users
#
#   Lending Club Loan Dataset: https://www.kaggle.com/wendykan/lending-club-loan-data
#
###############################################################################################################

from pysparkling import *
hc = H2OContext.getOrCreate(spark)
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Import Data
# https://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/docs-website/h2o-py/docs/h2o.html

input_schema =  {
                    "id":"string",
                    "member_id":"string",
                    "loan_amnt":"float",
                    "term_in_months":"int",
                    "interest_rate":"float",
                    "payment grade":"string",
                    "sub_grade":"string",
                    "employment_length":"int",
                    "home_owner":"int",
                    "income":"float",
                    "verified":"int",
                    "default":"int",
                    "purpose":"string",
                    "zip_code":"string",
                    "addr_state":"string",
                    "open_accts":"int",
                    "credit_debt":"float"
                }


loans = h2o.import_file(                \
        path="/loan_200k.csv",          \
        header=0,                       \
        sep=',',                        \
        col_names=column_names,         \
        col_types=column_types          \
        )

# Data Descriptive Info
loans.columns
loans.head()
loans.types
loans.summary()
loans.shape

loans["addr_state"].table()
loans["default"].table()

# Convert to and from Spark DataFrame
df_loans = hc.as_spark_frame(loans,)
back_to_h20_df = hc.as_h2o_frame(df_loans)

# H2O Change Data type
loans["default"].summary()
loans["default"] = loans["default"].asfactor()
loans["default"].summary()

# Split H2O data table into train test and ** validation ** datasets
training, testing, validation = loans.split_frame([0.70,0.20],seed=12345)

# Specify Target
target = "default"

# Specify Predictors
predictors = loans.names[:]
predictors.remove(target)
predictors.remove("id")
predictors.remove("member_id")

# Modeling
model_gbm = H2OGradientBoostingEstimator(ntrees=50, max_depth=6, learn_rate=0.1, distribution="bernoulli")
model_gbm.train(x=predictors, y=target, training_frame=training, validation_frame=validation)
predictions = model_gbm.predict(testing)
predictions.head()

# Simple Deep Learning - Predict Arrest
model_dl = H2ODeepLearningEstimator(variable_importances=True, loss="Automatic")
model_dl.train(x=predictors, y=target, training_frame=training, validation_frame=validation)
predictions = model_dl.predict(testing)
predictions.head()

# Save Model
model_path = h2o.save_model(model=model, path="/tmp/mymodel", force=True)
print model_path

# Load the model
# saved_model = h2o.load_model(model_path)


#ZEND
