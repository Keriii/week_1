import pandas as pd
from sqlalchemy import create_engine
import psycopg2


def connection():
    """
    connects to my pg database
    """
    conn = psycopg2.connect(dbname = 'Telecom',
                            user = 'postgres',
                            password = 'kerod53@',
                            host = 'localhost',
                            port = '5432')
    return conn

def table_to_sql(conn, table_name:str) -> pd.DataFrame:
    query = f'SELECT * FROM public.{table_name}'
    data = pd.read_sql_query(query, conn)

    return data
my_parameters = {'dbname': 'Telecom',
                'user': 'postgres',
                'password':'kerod53@',
                'host':'localhost',
                'port': '5432'}

def connection2(self, parameters: dict = my_parameters):
    """
    Connects to the PostgreSQL database. for a given paramters:
    connection_params is a dictionary that define the following:
    {
        'dbname': 'your_database_name',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'your_host',
        'port': 'your_port'
        }
    """
    try:
        conn = psycopg2.connect(**parameters)
        return conn
    except psycopg2.Error as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None

def get_dataframe_sql(self, conn, table_name) -> pd.DataFrame:
    query = f'SELECT * FROM public.{table_name}'
    data = pd.read_sql_query(query, conn)
    return data

def create_engine(self, connection_params: dict = my_parameters):
    """
    creates engine using sqlalchemy for a given paramters:
    """
    engine = create_engine("postgresql://postgres:kerod53@@localhost:5432/Telecom")
    return engine

def write_dataframe_to_table(self, df: pd.DataFrame, table_name: str,engine)->None:
    """
    Writes a pandas dataframe to a new table in the PostgreSQL database.
    """
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print(f"Dataframe successfully written to the '{table_name}' table.")

def update_table_by_appending(df, table_name, connection_params = my_parameters):
    """
    Appends a pandas dataframe to an existing PostgreSQL table.
    """
    engine = create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}")
    df.to_sql(table_name, engine, index=False, if_exists='append')
    print(f"Dataframe successfully appended to the '{table_name}' table.")

def delete_table(table_name, connection_params = my_parameters):
    """
    Deletes a table from the PostgreSQL database.
    """
    connection = connect_to_database(connection_params)
    if connection:
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        connection.commit()
        connection.close()
        print(f"Table '{table_name}' successfully deleted.")
    else:
        print("Error: Unable to connect to the database.")

def connect(self):
    conn = psycopg2.connect(dbname = 'Telecom',
                            user = 'postgres',
                            password = 'kerod53@',
                            host = 'localhost',
                            port = '5432')
    return conn

def get_data_sql(self, conn, table_name):
    query = f'SELECT * FROM public.{table_name}'
    data = pd.read_sql_query(query, conn)
    return data
