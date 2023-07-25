import os
import pandas as pd
import mysql.connector


class MySQLService:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_DATABASE")
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            self.cursor = self.conn.cursor()
            print("Connected to MySQL database")
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL database: {e}")
            raise

    def get_data(self, table_name, columns=None, where_clause=None):
        try:
            # If columns are not specified, fetch all columns (*)
            columns_str = "*" if columns is None else ", ".join(columns)

            # Build the SQL query with the optional WHERE clause
            query = f"SELECT {columns_str} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            self.cursor.execute(query)
            result = self.cursor.fetchall()

            # Convert the fetched data to a DataFrame
            column_names = [col[0] for col in self.cursor.description]
            df = pd.DataFrame(result, columns=column_names)

            return df
        except mysql.connector.Error as e:
            print(f"Error fetching data: {e}")
            return None

    def close(self):
        if self.conn and self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            print("Connection to MySQL database closed.")
