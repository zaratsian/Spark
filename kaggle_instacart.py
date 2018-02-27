

####################################################################################################################
#
#   Kaggle Instacart Competition
#
#   Predict which previously purchased products will be in a user’s next order.
#
####################################################################################################################

'''

Download Data from: https://www.kaggle.com/c/instacart-market-basket-analysis/data

scp -i ~/.ssh/field.pem ~/Downloads/aisles.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.
scp -i ~/.ssh/field.pem ~/Downloads/departments.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.
scp -i ~/.ssh/field.pem ~/Downloads/order_products__prior.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.
scp -i ~/.ssh/field.pem ~/Downloads/order_products__train.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.
scp -i ~/.ssh/field.pem ~/Downloads/orders.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.
scp -i ~/.ssh/field.pem ~/Downloads/products.csv.zip centos@dzaratsian4.field.hortonworks.com:/tmp/.

ssh -i ~/.ssh/field.pem dzaratsian4.field.hortonworks.com

cd /tmp
unzip aisles.csv.zip
unzip departments.csv.zip
unzip order_products__prior.csv.zip
unzip order_products__train.csv.zip
unzip orders.csv.zip
unzip products.csv.zip

sudo su
su hdfs

hadoop fs -mkdir /data/
hadoop fs -mkdir /data/instacart
hadoop fs -mkdir /data/instacart/aisles
hadoop fs -mkdir /data/instacart/departments
hadoop fs -mkdir /data/instacart/order_products__prior
hadoop fs -mkdir /data/instacart/order_products__train
hadoop fs -mkdir /data/instacart/orders
hadoop fs -mkdir /data/instacart/products

hadoop fs -put /tmp/aisles.csv /data/instacart/aisles/.
hadoop fs -put /tmp/departments.csv /data/instacart/departments/.
hadoop fs -put /tmp/order_products__prior.csv /data/instacart/order_products__prior/.
hadoop fs -put /tmp/order_products__train.csv /data/instacart/order_products__train/.
hadoop fs -put /tmp/orders.csv /data/instacart/orders/.
hadoop fs -put /tmp/products.csv /data/instacart/products/.


'''


from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import col, collect_set


spark = SparkSession \
    .builder \
    .appName("PySpark Instacart") \
    .enableHiveSupport() \
    .getOrCreate()

####################################################################################################################
#
#   Load Data
#
####################################################################################################################

df_aisles = spark.read.csv('hdfs:///data/instacart/aisles/aisles.csv', inferSchema=True, header=True)
df_aisles.show(10,False)

df_departments = spark.read.csv('hdfs:///data/instacart/departments/departments.csv', inferSchema=True, header=True)
df_departments.show(10,False)

df_order_products__prior = spark.read.csv('hdfs:///data/instacart/order_products__prior/order_products__prior.csv', inferSchema=True, header=True)
df_order_products__prior.show(10,False)

df_order_products__train = spark.read.csv('hdfs:///data/instacart/order_products__train/order_products__train.csv', inferSchema=True, header=True)
df_order_products__train.show(10,False)

df_orders = spark.read.csv('hdfs:///data/instacart/orders/orders.csv', inferSchema=True, header=True)
df_orders.show(10,False)

df_products = spark.read.csv('hdfs:///data/instacart/products/products.csv', inferSchema=True, header=True)
df_products.show(10,False)

####################################################################################################################
#
#   Descriptive Stats
#
####################################################################################################################

def descriptive_stats(df):
    print('*'*100)
    print('\nRow Count: ' + str(df.count()) + '\n')
    print('\nColumns and Data Types:')
    for column,dtype in df.dtypes:
        print('\t' + str(column) + ', ' + str(dtype))
    print('\n')
    categorical_cols = [item[0] for item in df.dtypes if item[1]=='string']
    for categorical_col in categorical_cols:
        df.groupBy(categorical_col).count().orderBy(col('count').desc()).show()
    print('\n')
    df.describe().show(10,False)
    print('*'*100)

print('\n\nDescriptive Stats for: df_aisles')
descriptive_stats(df_aisles)

print('\n\nDescriptive Stats for: df_departments')
descriptive_stats(df_departments)

print('\n\nDescriptive Stats for: df_order_products__train')
descriptive_stats(df_order_products__train)

print('\n\nDescriptive Stats for: df_orders')
descriptive_stats(df_orders)

print('\n\nDescriptive Stats for: df_products')
descriptive_stats(df_products)

####################################################################################################################
#
#   Drop NAs
#
####################################################################################################################

df_aisles.count()
df_aisles = df_aisles.na.drop()
df_aisles.count()

df_departments.count()
df_departments = df_departments.na.drop()
df_departments.count()

df_order_products__train.count()
df_order_products__train = df_order_products__train.na.drop()
df_order_products__train.count()

df_orders.count()
df_orders = df_orders.na.drop()
df_orders.count()

df_products.count()
df_products = df_products.na.drop()
df_products.count()


####################################################################################################################
#
#   Joins
#
####################################################################################################################

df_joined1 = df_order_products__train.join(df_orders, df_order_products__train.order_id == df_orders.order_id, 'left') \
                .drop(df_orders.order_id)

df_joined1.show(20,False)

df_order_products__train.count()
df_orders.count()
df_joined1.count()

# Descriptive Stats
df_joined1.describe().show(10,False)


####################################################################################################################
#
#   Model Prep
#
####################################################################################################################

order_list = df_order_products__train \
    .select(['order_id','product_id']) \
    .groupby("order_id") \
    .agg(collect_set("product_id")) \
    .withColumnRenamed('collect_set(product_id)','product_set')

order_list.show(20,False)


#(training, test) = df_joined1.randomSplit([0.8, 0.2])

####################################################################################################################
#
#   Train Model
#
####################################################################################################################

fpGrowth = FPGrowth(itemsCol="product_set", minSupport=0.01, minConfidence=0.05)
model    = fpGrowth.fit(order_list)

# Display frequent itemsets.
model.freqItemsets.show(20,False)

# Display generated association rules.
association = model.associationRules \
                   .withColumn('antecedent_value', col('antecedent')[0]) \
                   .withColumn('consequent_value', col('consequent')[0])

association.show(20,False)

association.count()
df_products.count()
join_assoc1 = association.join(df_products.select(['product_id','product_name']), association.antecedent_value == df_products.product_id, 'left').drop('product_id').withColumnRenamed('product_name','antecedent_name')
join_assoc2 = join_assoc1.join(df_products.select(['product_id','product_name']), join_assoc1.consequent_value == df_products.product_id, 'left').drop('product_id').withColumnRenamed('product_name','consequent_name')
join_assoc1.count()
join_assoc2.count()
join_assoc2.show(100,False)

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(order_list).show(20,False)













#ZEND