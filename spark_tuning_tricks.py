

# ...in progress...


# Get Spark Configs
def get_spark_configs():
    configs = {}
    for config in sc._conf.getAll():
        configs[config[0]] = config[1]
    return configs


# Add python package to all executors
def import_py_packages(executor_num):
  import textblob
  return executor_num

rdd = sc.parallelize([1, 2, 3])
rdd.map(lambda x: import_py_packages(x))
rdd.collect()


#ZEND
