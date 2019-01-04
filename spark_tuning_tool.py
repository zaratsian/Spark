
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
import argparse


if __name__ == "__main__":
    
    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_count",     required=True,  type=int, help="Number of workers")
    ap.add_argument("--node_cores",     required=True,  type=int, help="Cores per node")
    ap.add_argument("--node_ram",       required=True,  type=int, help="Node RAM (in GBs)")
    ap.add_argument("--executor_cores", required=False, type=int, help="Number of cores per executor")
    ap.add_argument("--yarn_memory",    required=False, type=int, help="YARN memory (in bytes)")
    ap.add_argument("--yarn_vcores",    required=False, type=int, help="YARN vcores")
    args = vars(ap.parse_args())
    
    
    try:
        executor_cores   = int(args['executor_cores'])
    except:
        executor_cores = 5  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
    
    try:
        yarn_nodemanager_resource_memory_mb  = int(args['yarn_memory'])
    except:
        yarn_nodemanager_resource_memory_mb  = (args['node_ram'] - 4) * 1024
    
    try:
        yarn_nodemanager_resource_cpu_vcores = int(args['yarn_vcores'])
    except:
        yarn_nodemanager_resource_cpu_vcores = (args['node_cores'] - 2)
    
    
    total_cores = args['node_count'] * args['node_cores']
    total_ram   = args['node_count'] * args['node_ram']
    
    
    executor_cores     = executor_cores  # ~5 or less typically and ideally is a divisor of yarn.nodemanager.resource.cpu-vcores)
    executors_per_node = yarn_nodemanager_resource_cpu_vcores / executor_cores 
    executor_memory    = round( ((yarn_nodemanager_resource_memory_mb / executors_per_node))/1024 - 2)  # Subtract 2GB for buffer/extra space
    num_executors      = round((args['node_count'] * executors_per_node ) - 1) # Subtract 1 for Driver, since Driver will consume 1 of exector slots
    driver_cores       = executor_cores
    driver_memory      = executor_memory
    
    
    # Output Summary
    print('\n\n####################################################################\n' + \
        '\nNode Count:                      ' + str(args['node_count']) + \
        '\nNode Cores:                      ' + str(args['node_cores']) + \
        '\nNode RAM:                        ' + str(args['node_ram']) + ' GB' \
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
        '\n\n####################################################################\n')
    

# TODO
# Add unused core count per node
# Add unused memory per node


#ZEND
