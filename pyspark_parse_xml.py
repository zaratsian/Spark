

#/spark/bin/pyspark --packages com.databricks:spark-xml_2.11:0.4.1


from pyspark.sql.functions import *
from pyspark.sql.types import *

df = spark.read.format('com.databricks.spark.xml').options(rowTag='record').load('/tmp/data.xml')

df.show()

def parse_list(column, index):
    return column[index]

udf_parse_list = udf(parse_list, StringType())

df.withColumn('size', udf_parse_list(df.transaction, lit(0)))  \
  .withColumn('value', udf_parse_list(df.transaction, lit(1))) \
  .show()




# data.xml (below)

'''
<?xml version="1.0"?>
<records>
  <record>
    <id>1000</id>
    <transaction>
        <value>10</value>
        <size>S</size>
    </transaction>
  </record>
  <record>
    <id>2000</id>
    <transaction>
        <value>20</value>
        <size>M</size>
    </transaction>
  </record>
  <record>
    <id>3000</id>
    <transaction>
        <value>30</value>
        <size>L</size>
    </transaction>
  </record>
  <record>
    <id>4000</id>
    <transaction>
        <value>40</value>
        <size>XL</size>
    </transaction>
  </record>
  <record>
    <id>5000</id>
    <transaction>
        <value>50</value>
        <size>XXL</size>
    </transaction>
  </record>
</records>
'''



#ZEND
