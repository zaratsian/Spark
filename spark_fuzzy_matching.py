

#####################################################################################################################
#
#   PySpark Fuzzy Matching
#
#   Soundex         http://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#pyspark.sql.functions.soundex
#   Levenshtein     
#   https://medium.com/@mrpowers/fuzzy-matching-in-spark-with-soundex-and-levenshtein-distance-6749f5af8f28
#
#####################################################################################################################


from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import soundex, concat, levenshtein, unix_timestamp, from_unixtime
import datetime,time


#####################################################################################################################
#
#   Fuzzy Matching within a single table (using Soundex)
#
#####################################################################################################################

# Generate Dataframe for testing
df = spark.createDataFrame(
    [
        (['dan',    'ocean',        'nc', '05/25/1983']),
        (['daniel', 'ocean',        'nc', '05/25/1983']),
        (['danny',  'ocean',        'nc', '05/26/1983']),
        (['danny',  'ocen',         'nc', '05/26/1983']),
        (['danny',  'oceans11',     'nc', '04/26/1982']),
        (['tess',   'ocean',        'nc', '02/10/1988']),
        (['john',   'smith',        'ca', '01/30/1980']),
        (['john',   'smith',        'ca', '09/30/1981'])
    ], 
    ['firstname','lastname','state','dob']
    )

df.show(10,False)

# Step 1: Resolve any known name aliases, states, etc (i.e. dan, daniel, danny)
# For this POC code, I chose not to include this step since it's straight-forward to add a dictionary for matching and resolving known aliases.

# Step 2: Clean & Process other fields (ie. convert dates)
df = df.withColumn('dob_formatted', from_unixtime(unix_timestamp('dob', 'MM/dd/yyyy'), format='yyyy_MMMMMMMM_dd') )

# Step 3: Concat all relevant matching fields
df = df.withColumn('record_uid', concat(df.state, df.dob_formatted, df.firstname, df.lastname))

# Step 4: Soundex encoding (score record_uid for similarities)
df.withColumn('score_soundex', soundex(df.record_uid)).show(10,False)


#####################################################################################################################
#
#   Fuzzy Matching Join using Levenshtein
#
#####################################################################################################################

# Generate Dataframe for testing
df = spark.createDataFrame(
    [
        (['dan',    'ocean',        'nc', '05/25/1983']),
        (['daniel', 'ocean',        'nc', '05/25/1983']),
        (['danny',  'ocean',        'nc', '05/26/1983']),
        (['danny',  'ocen',         'nc', '05/26/1983']),
        (['danny',  'oceans11',     'nc', '04/26/1982']),
        (['tess',   'ocean',        'nc', '02/10/1988']),
        (['john',   'smith',        'ca', '01/30/1980']),
        (['john',   'smith',        'ca', '09/30/1981'])
    ], 
    ['firstname','lastname','state','dob']
    )

df.show(10,False)

# Generate Dataframe 2 for testing
df2 = spark.createDataFrame(
    [
        (['dan',    'ocean',        '05/25/1983',   'medical code AAA']),
        (['danny',  'oceans11',     '04/26/1982',   'medical code BBB']),
        (['tess',   'ocean',        '02/10/1988',   'medical code CCC']),
        (['john',   'smith',        '01/30/1980',   'medical code DDD']),
        (['john',   'smith',        '09/30/1981',   'medical code EEE'])
    ], 
    ['firstname','lastname','dob','medical_code']
    )

df2.show(10,False)

# 1) Concat relevant fields used for fuzzy matching into a field called join_id
# 2) Apply levenshtein distance (which generates a score)
# 3) Use this score as a join criteria
# 4) Join on join_id

joinedDF = df.join(df2,
            levenshtein( concat(df.dob,df.firstname,df.lastname), concat(df2.dob,df2.firstname,df2.lastname) ) < 5,
            how='left_outer'
            )

joinedDF.show(10,False)




#ZEND