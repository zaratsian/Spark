
######################################################################################################
#
#   Parse Zeppelin Notebook (in .json format) into a text formatted file
#
#   Input:   yourZeppelin.json (Zeppelin Notebook)
#   Ouptut:  yourZeppelin.txt  (Text equivalent, which can be a python, spark, R, bash, ect script)
#
#   Usage:
#       python zeppelin_parse_to_text.py <your_zeppelin_notebook.json> <output_filename>
#
######################################################################################################

import sys,os
import json

try:
    input_file = sys.argv[1]
    output_file   = sys.argv[2]
    zeppelin_file = open(input_file,'rb')
    zeppelin_data = zeppelin_file.read().decode('utf-8-sig')
    zeppelin_json = json.loads(zeppelin_data.encode('utf-8'),'utf-8')
    zeppelin_file.close()
except:
    print '\n\n[ ERROR ] There is an issue reading in your Zeppelin Notebook. Make sure that your notebook is in .json format. Also make sure you has specified your input and output files\n\nUsage: python zeppelin_parse_to_text.py <your_zeppelin_notebook.json> <output_filename>\n'
    sys.exit()

zeppelin_name  = zeppelin_json['name']
zeppelin_id    = zeppelin_json['id']
zeppelin_paras = zeppelin_json['paragraphs']

output_results = open(output_file,'wb')

for para in zeppelin_paras:
    output_results.write(re.sub('^\%','#%',para['text']))
    output_results.write('\r\n\r\n')

output_results.close()


#ZEND
