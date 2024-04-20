import os

import mysql.connector
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


class MySQLService:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_DATABASE")

    def connect(self):
        try:
            return mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
            )
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL database: {e}")
            raise

    def execute_query(self, query):
        db = self.connect()
        cursor = db.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        # Convert the fetched data to a DataFrame
        column_names = [col[0] for col in cursor.description]
        df = pd.DataFrame(result, columns=column_names)

        db.close()

        return df

    def get_data(self, table_name, columns=None, where_clause=None, order_by_clause=None):
        try:
            # If columns are not specified, fetch all columns (*)
            columns_str = "*" if columns is None else ", ".join(columns)

            # Build the SQL query with the optional WHERE clause
            query = f"SELECT {columns_str} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            if order_by_clause:
                query += f" ORDER BY {order_by_clause}"

            df = self.execute_query(query)

            return df
        except mysql.connector.Error as e:
            print(f"Error fetching data: {e}")
            return None
