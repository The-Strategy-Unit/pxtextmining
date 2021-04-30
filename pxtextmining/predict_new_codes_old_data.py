import re
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import json  # Just using this to give a prettier result
import joblib

# https://instructobit.com/tutorial/130/Python-MySQL-executing-multi-query-.sql-files
db = mysql.connector.connect(option_files="suce_my.conf", use_pure=True)
with db.cursor(buffered=True) as cursor:
    with open('notts_data_up_to_2020_10_01.sql', 'r') as sql_file:
        result_iterator = cursor.execute(sql_file.read(), multi=True)
        for res in result_iterator:
            print("Running query: ", res)
            if res.with_rows:
                fetch_result = res.fetchall()
                print(json.dumps(fetch_result, indent=4))
            elif res.rowcount > 0:
                print(f"Affected {res.rowcount} rows")
        col_names = res.column_names
notts_old_data = pd.DataFrame(fetch_result)
notts_old_data.columns = col_names

pipe = joblib.load('results for label/test_pipeline.sav')
notts_old_data = notts_old_data.rename(columns={'feedback': 'predictor'})
notts_old_data['predictions'] = pipe.predict(notts_old_data[['predictor']])
notts_old_data = notts_old_data.rename(columns={'predictor': 'feedback'})

# ====== Write results to database ====== #
# Pull database name & host and user credentials from my.conf file
conf = open('my.conf').readlines()
conf.pop(0)
for i in range(len(conf)):
    match = re.search('=(.*)', conf[i])
    conf[i] = match.group(1).strip()

# Connect to mysql by providing a sqlachemy engine
engine = create_engine(
    "mysql+mysqlconnector://" + conf[2] + ":" + conf[3] + "@" + conf[0] + "/" + conf[1],
    echo=False)

# Write results to database
print("Writing to database...")
notts_old_data.to_sql(name="predictions_notts_old_data_new_codes", con=engine, if_exists="replace", index=False)