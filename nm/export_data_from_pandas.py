import pandas as pd
import sqlalchemy
from config import mysql_url

# Create the engine to connect to a PostgreSQL database
engine = sqlalchemy.create_engine(mysql_url)

# Read data from the SQL Table
data = pd.read_csv('superstore.csv')

# Print first few rows of the dataframe
print(data.head())

#######################################################################################

# Write the data into a table in PostgreSQL Database

# noinspection PyPep8
data.to_sql(name = 'superstore',
            con = engine,
            if_exists='append',
            index = False,
            chunksize = 1000,
            dtype = {
                "Row_ID" : sqlalchemy.types.Integer,
                "Order_ID" : sqlalchemy.types.Text,
                "Order_Date" : sqlalchemy.types.DateTime,
                "Ship_Date" : sqlalchemy.types.DateTime,
                "Sales" : sqlalchemy.types.Numeric,
                "Quantity" : sqlalchemy.types.Integer,
                "Discount" : sqlalchemy.types.Numeric,
                "Profit" : sqlalchemy.types.Numeric,
            }
)