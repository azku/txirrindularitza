import re
import mysql.connector
from mysql.connector import Error


ARGAZKIEN_ERROA = "/home/ir_inf/data/wpcf7-files/out"

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")


def get_picture_metadata():
    connection = create_connection("fptxurdinaga.in", "txirrindularia", "txirrindularia_2024_2025","txirrindularitza_db")

    select_comments = "select post_content from wp_posts;"
    posts = execute_read_query(connection, select_comments)

    e1 = re.compile(r"[\d]{2,4}/[\d]{2,4}/[\d]{2,4}")
    e2 = re.compile(r"\d{1,2}:\d{1,2}")
    e3 = re.compile(r"picture_.*\.jpg$")
    hizt = {}
    for p in posts:
        if "picture_" in p[0]:
            l = p[0].splitlines()
            match1 = re.search(e1,l[1])
            match2 = re.search(e2,l[2])
            if match1 and match2:
                for pics in l:
                    match3 = re.search(e3, pics)
                    #gehitu hiztegira
                    if match3:
                        hizt[match3.group(0)] = {"id":l[0] ,"date":match1.group(0), "time":match2.group(0), "route": l[3]}

    connection.close()
    return hizt
