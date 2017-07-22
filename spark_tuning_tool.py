
##########################################################################################
#
#   Spark Tuning Tool
#
#   Dan Zaratsian
#
#   Usage: spark_tuning_tool.py --node_count=20 --node_cores=16 --node_ram=64 [ --executor_cores=5 ] [ --yarn_memory=8192 ] [ --yarn_vcores=8 ]
#
##########################################################################################

import sys
import getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], 'x', ['node_count=', 'node_cores=', 'node_ram=', 'executor_cores=', 'yarn_memory=', 'yarn_vcores='])
    
    node_count = [int(opt[1]) for opt in opts if opt[0]=='--node_count'][0]
    node_cores = [int(opt[1]) for opt in opts if opt[0]=='--node_cores'][0]
    node_ram   = [int(opt[1]) for opt in opts if opt[0]=='--node_ram'][0]
    try:
        executor_cores   = [int(opt[1]) for opt in opts if opt[0]=='--executor_cores'][0]
    except:
        executor_cores = 5  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
    
    try:
        yarn_nodemanager_resource_memory_mb  = [int(opt[1]) for opt in opts if opt[0]=='--yarn_memory'][0]
    except:
        yarn_nodemanager_resource_memory_mb  = (node_ram - 4) * 1024
    
    try:
        yarn_nodemanager_resource_cpu_vcores = [int(opt[1]) for opt in opts if opt[0]=='--yarn_vcores'][0]
    except:
        yarn_nodemanager_resource_cpu_vcores = (node_cores - 2)
    
except:
    print '\n'
    print '\n[ USAGE ] spark_tuning_tool.py --node_count=<number> --node_cores=<number> --node_ram=<number_in_GBs>'
    print '\n[ USAGE ] spark_tuning_tool.py --node_count=<number> --node_cores=<number> --node_ram=<number_in_GBs> [ --executor_cores=5 ] [ --yarn_memory=8192 ] [ --yarn_vcores=8 ]\n\n'
    sys.exit(1)


total_cores = node_count * node_cores
total_ram   = node_count * node_ram


executor_cores     = executor_cores  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
executors_per_node = yarn_nodemanager_resource_cpu_vcores / executor_cores 
executor_memory    = ((yarn_nodemanager_resource_memory_mb / executors_per_node))/1024 - 2  # Subtract 2GB for buffer/extra space
num_executors      = (node_count * executors_per_node ) - 1 # Subtract 1 for Driver, since Driver will consume 1 of exector slots
driver_cores       = executor_cores
driver_memory      = executor_memory


# Output Summary
print '\n\n####################################################################\n' + \
    '\nNode Count:                      ' + str(node_count) + \
    '\nNode Cores:                      ' + str(node_cores) + \
    '\nNode RAM:                        ' + str(node_ram) + ' GB' \
    '\n' + \
    '\nTotal Cores:                     ' + str(total_cores) + \
    '\nTotal RAM:                       ' + str(total_ram) + ' GB' \
    '\n' + \
    '\nYARN Nodemanager RAM (mb):       ' + str(yarn_nodemanager_resource_memory_mb) + \
    '\nYARN Nodemanager CPU Vcores:     ' + str(yarn_nodemanager_resource_cpu_vcores) + \
    '\n' + \
    '\nexecutors_per_node:              ' + str(executors_per_node) + \
    '\n' + \
    '\n--executor-cores:                ' + str(executor_cores) + \
    '\n--executor-memory:               ' + str(executor_memory) + ' GB' \
    '\n--num-executors:                 ' + str(num_executors) + \
    '\n\n' + \
    './bin/spark-submit --master yarn --deploy-mode cluster' + ' --driver-cores ' + str(driver_cores) + ' --driver-memory ' + str(driver_memory) + 'G' + ' --executor-memory ' + str(executor_memory) + 'G' + ' --num-executors ' + str(num_executors) + ' --executor-cores ' + str(executor_cores) + \
    '\n\n' + \
    '/usr/hdp/current/spark2-client/bin/pyspark --master yarn --deploy-mode client' + ' --driver-cores ' + str(driver_cores) + ' --driver-memory ' + str(driver_memory) + 'G' + ' --executor-memory ' + str(executor_memory) + 'G' + ' --num-executors ' + str(num_executors) + ' --executor-cores ' + str(executor_cores) + \
    '\n\n####################################################################\n' 



# Add unused core count per node
# Add unused memory per node


#ZEND
