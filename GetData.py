"""
Author: Abulele Mditshwa
Student No: Junior Data Scientist
email: abulele@capeai.com
The objective of this code is to extract data from a SQL database.
"""


# all the neccessary visualiztion libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import traceback
import calendar

#SQL connection
import mysql.connector  # PUP.Pargo uses MYSQL
import psycopg2         # Simba uses Postgres
from mysql.connector import MySQLConnection, Error



############### Please take note of these variable as they will change time to time
PickupPoint = "Lewis" # choose which pickup point we want, This will allow us to create a report for any desired PUP.
month = "01"
year = "2020"
previous_months = 24

#####################################################################################Connects the Simba database###################################################
def Simba_database(command):
    """
    This method connects to the Simba database, and returns an error if you can't.
    It takes a argument, which is SQL command to the database
    """
    user_pargo = "pargo_employee"
    host_pargo = "31.3.103.66"
    password_pargo = "146J4XY8dQYqZrVcUhaY"
    database_pargo = "pargo"
    port_pargo = "9364"

    try:
        # Connect to database
        connection = psycopg2.connect(user=user_pargo,
                                      password=password_pargo,
                                      host=host_pargo,
                                      port=port_pargo,
                                      database=database_pargo)
        cursor = connection.cursor()
        # Try to send command. If not, print the SQL error
        cursor.execute(str(command))
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        return results, column_names
    except psycopg2.DatabaseError as e:
        # Message stating export unsuccessful.
        print("Database error dumbass !", e)
        traceback.print_exc()

    except TypeError as e:
        print("TypeError occured")
        print(e)
    finally:
        # Close database connection.
        connection.close()



##########################################################################Connects to the Pargo database.################################################################
def Pargo_database(command):
    """
    This method connects to the Pargo database and sends an error if you can't.
    It takes a SQL command as an argument to perform desired operations to the Pargo Database.
    """
    conn = None
    user_pargo = "keesvanbezouw"            #Username
    host_pargo = "185.21.189.94"            #where the DB is hosted
    password_pargo = "ucMGmjKyMz6XJUA9"     # Password Pargo
    database_pargo = "pargo_db"             # database name.
    try:
        conn = mysql.connector.connect(host=host_pargo,
                                       database=database_pargo,
                                       user=user_pargo,
                                       password=password_pargo)
        if conn.is_connected():
            cursor = conn.cursor() # to perform SQL operations.
            cursor.execute(str(command))

            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return results,column_names



    except Error as e:
        print("Error connecting to the database",e)

    finally:
        cursor.close()
        conn.close()
###########################################################################################################################################

def pargoGetOrderStatus():
    """
    This function gets the order status table in the PUP.pargo database, and returns the values as a Dataframe.
    """

    sqlCommandCollected = """
    SELECT orders_status.ordersid, MAX(orders_status.order_status_collected) AS order_status_collected
    FROM orders_status
    INNER JOIN orders ON orders_status.ordersid = orders.ordersid
    WHERE orders_status.order_status_collected IS NOT NULL """ + "AND MONTH(orders_status.order_status_collected) = " + str(month) + " AND YEAR(orders_status.order_status_collected) = " + str(year) + " GROUP BY ordersid"

    ## The SQL query selects two columns from the order_status table
    ## Joins two tables where the ordersid matches
    ## we don't want parcels that are not collected. (NULL values)

    try:
        results, column_names = Pargo_database(sqlCommandCollected)
        # function returns two values
        order_status = pd.DataFrame(results)
        order_status.columns = [column_names]
    except ValueError:
        order_status = pd.DataFrame(columns=["ordersid","order_status_collected"])
        # if the dataframe is empty.v
        return order_status

    except Error as e:
        print("Database error you ass :( ",e)

    return order_status

def pargoGetPargoPoints(PickupPoints):

    """
    Function that takes pargopoints and few other columns to match with the orders table in the PUP.pargo database.
    """
    sqlCommandPUPS = """
        SELECT pargopoint.pargopointid, pargopoint.store_name, pargopoint.type_ppp,pargopoint.latitude, pargopoint.longitude,
        pargopoint.active, pargopoint.province, orders.ordersid
        FROM pargopoint
        INNER JOIN orders ON pargopoint.pargopointid = orders.pargopointid
        WHERE pargopoint.store_name LIKE """ + "'%" +PickupPoint+ "%' " + "AND pargopoint.active = 1"

    # This selects data from Pargopoints
        # Selects 7 columns
        # does an inner join with orders table where it matches the ordersid column
        # we only take the active PargoPoints and ignore the rest.

    try:
        results, column_names = Pargo_database(sqlCommandPUPS)
        # function returns two values
        results_PUPS = pd.DataFrame(results)
        results_PUPS.columns = [column_names]

    except ValueError:
        results_PUPS = pd.DataFrame(columns=["store_name","active","type_ppp","pargopointid","ordersid"])
        return results_PUPS

    return results_PUPS

## So far the two functions  gets  the orders that have been collected from the PUP.Pargo and they match it with a Particular Pickup point of your choice.
# stores the dataFrame in a variables

pargo_points = pargoGetPargoPoints(PickupPoint)
pargo_orderstatus = pargoGetOrderStatus()

pargo_points.columns = pargo_points.columns.get_level_values(0)
pargo_orderstatus.columns = pargo_orderstatus.columns.get_level_values(0)
# Now we Merge the two DataFrames together.
# Merge the Pargo Points and Order Status. based on the ordersid column.
# This dataframe will show us the orders that have been collected and at which Pickup Point
mergedOrderCollections = pd.merge(pargo_points,pargo_orderstatus, on="ordersid")
#print(mergedOrderCollections.head())

# Now we get data from the Simba database

def getOrderProcess(PickupPoint):
    """
    1. Gets the order_process table from simba and just selects table that define how the process of the order goes.
    """

    sqlOrderProcess ="""
    SELECT order_processes.process_type_id,order_processes.to_id,order_processes.to_name,order_processes.to_type, order_processes.to_province,orders.first_order_process_id AS order_process_id
    FROM order_processes
    INNER JOIN orders ON order_processes.uuid = orders.first_order_process_id AND to_name LIKE """ +  "'%" +str(PickupPoint)+ "%'" + """ 
    INNER JOIN pickup_points ON order_processes.to_id = pickup_points.uuid AND pickup_points.is_active = True
    INNER JOIN process_types ON order_processes.process_type_id = process_types.uuid WHERE process_types.name = 'w2p'
     """

    try:
        results, column_names = Simba_database(sqlOrderProcess)
        results = pd.DataFrame(results)
        results.columns = [column_names]

    except ValueError:
        print("You got an empty dataframe")

    except Error as e:
        print("Database error dumb ass :(",e)


    return results


#check = getOrderProcess(PickupPoint)
#print(check.head()

def getTrackingInfo(month, year):
    """
    1. gets the tracking info the parcels.
    """

    sqlTrackingInfo = """
    SELECT order_process_tracking_info.order_process_id ,order_process_tracking_info.created_at AS timestamp,
    process_types_stages.name AS process_type_stage_name
    FROM order_process_tracking_info 
    INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id 
    INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id 
    INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id 
    INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id 
    WHERE (process_types_stages.name = 'completed') and order_process_tracking_info.order_process_id IN ( 
        SELECT order_process_tracking_info.order_process_id 
            FROM order_process_tracking_info 
            INNER JOIN process_types on process_types.uuid = order_process_tracking_info.process_type_id 
            INNER JOIN order_processes on order_processes.uuid = order_process_tracking_info.order_process_id 
            INNER JOIN process_types_stages on process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id 
            INNER JOIN pickup_points on pickup_points.uuid = order_processes.to_id 
            WHERE process_types.name = 'w2p' 
            AND EXTRACT(MONTH FROM order_process_tracking_info.created_at) = """+ month +"""
            AND EXTRACT(YEAR FROM order_process_tracking_info.created_at) = """ + year + ");"

    try:
        results, columns = Simba_database(sqlTrackingInfo)
        results = pd.DataFrame(results)
        results.columns = [columns]

    except ValueError:
        print("You got an empty dataframe")

    except Error as e:
        print("You got database error :(", e)

    return results


#order_process_df = getOrderProcess(PickupPoint)
order_tracking_info_df = getTrackingInfo(month,year)
print(order_tracking_info_df.head())

#order_process_df.columns = order_process_df.columns.get_level_values(0)
#order_tracking_info_df.columns = order_tracking_info_df.columns.get_level_values(0)

#simbaProcess = pd.merge(order_process_df,order_tracking_info_df, on="order_process_id")
#print(simbaProcess.head())

