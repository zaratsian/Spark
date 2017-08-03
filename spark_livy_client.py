

##################################################################################################
#
#   Livy Server - Spark Example (REST API)
#
##################################################################################################


import requests
import json
import datetime,time
import cloudpickle


host    = 'dzaratsian5.field.hortonworks.com'
port    = '8999'
url     = 'http://' + str(host) +':'+ str(port)
headers = {'Content-Type': 'application/json', 'X-Requested-By': 'spark'}



def livy_get_sessions():
    req_sessions     = requests.get(url + '/sessions', headers=headers)
    sessions_payload = json.loads(req_sessions.content)
    return sessions_payload



def livy_delete_all_sessions():
    req_sessions     = requests.get(url + '/sessions', headers=headers)
    sessions_payload = json.loads(req_sessions.content)
    print '[ INFO ] Total Sessions:     ' + str(sessions_payload['total'])
    for session in sessions_payload['sessions']:
        session_id = session['id']
        delete_session = requests.delete(url + '/sessions/' + str(session_id), headers=headers)
        print '[ INFO ] Deleted Session ID: ' + str(session_id)



def livy_initialize_spark_session():
    data = {'kind': 'pyspark'}
    spark_session = requests.post(url + '/sessions', data=json.dumps(data), headers=headers)
    if spark_session.status_code == 201:
        print '[ INFO ] Status Code: ' + str(spark_session.status_code)
        print '[ INFO ] State:       ' + str(spark_session.json()['state'])
        print '[ INFO ] Payload:     ' + str(spark_session.content)
        session_id  = spark_session.json()['id']
        session_url = url + spark_session.headers['location']
    else:
        print '[ INFO ] Status Code: ' + str(spark_session.status_code)
    return spark_session



if __name__ == "__main__":
    
    spark_session = livy_initialize_spark_session()
    session_url   = url + spark_session.headers['location']
    check_session = requests.get(session_url, headers=headers)
    print '[ INFO ] Session ' + str(check_session.json()['id']) + ' state: ' + str(check_session.json()['state'])
    
    data = {'code': 'print spark.range(0,10).count()'}
    data = {'code': '1 + 1'}
    
    statement_url = session_url + '/statements'
    submit_code   = requests.post(statement_url, data=json.dumps(data), headers=headers)
    submit_code.status_code
    submit_code.json()
    
    statement_url = url + submit_code.headers['location']
    code_response = requests.get(statement_url, headers=headers)
    code_response.status_code
    code_response.json()





#ZEND
