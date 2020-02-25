"""
This file gets access to Pargo's database an runs some queries nan
"""

#Supplier dict

supplier_dict_pup = {'Digital Planet': ['sup492', 'sup385'], 'HomeChoice': [''], 'TFG': ['sup392'],
                     'Cape Union Mart': ['sup695', 'sup850', 'sup848', 'sup852', 'sup851'],
                     'Faithful to Nature': ['sup702'], 'Clicks' : [], 'Yoco' : ['sup478'],
                     'Raru': ['sup7'], 'Loot': [''], 'OneDayOnly':['sup6'], 'Wellness Warehouse':['sup860']}


supplier_dict_simba = {'Digital Planet': [''], 'HomeChoice': ['sup1502'], 'TFG': ['sup1512', 'sup1513'],
                     'Cape Union Mart':['sup1531', 'sup1535', 'sup1533', 'sup1532', 'sup1534'],
                     'Faithful to Nature': ['sup1504'], 'Clicks' : ['1506'], 'Yoco' : [''],
                     'Raru': [''], 'Loot': ['sup1556'], 'OneDayOnly':['sup6'], 'Wellness Warehouse':['sup1508']}


############################################################################################################################
# dictionary of suppliers.

#HyperParameters declaration
company = "Cape Union Mart" #Company to consider. The unlimited works, TFG WL works (sales are only starting from sept. ). TFG has very little.
supplier_list_pup_pargo =  supplier_dict_pup[str(company)]#THIS HAS TO BE A LIST aka circumvented by brackets [].


supplier_list_simba  = supplier_dict_simba[str(company)] #THIS HAS TO BE A LIST aka circumvented by brackets [].

month_nr = '01'   #Month to consider
year_nr = '2020'   #Year to consider
nr_months_back = 24 # Plot historial parcel sell nr from prev nr_months_back




import csv
import os
import psycopg2
import pandas as pd
import numpy as np
import sys
import traceback
import calendar
import matplotlib.pyplot as plt
import datetime
import random
from datetime import time
import imgkit
import six
import math
#Connect to old database
from mysql.connector import MySQLConnection, Error
#from python_mysql_dbconfig import read_db_config
import mysql.connector
import pyglet

#Import write to PDF modules
from reportlab.pdfgen import canvas 
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import landscape
from reportlab.platypus import Image
from reportlab.rl_config import defaultPageSize

from reportlab.rl_config import defaultPageSize

from reportlab.pdfbase.pdfmetrics import stringWidth

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from io import StringIO
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from PIL import Image
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics import renderPDF, renderPM


from textwrap import wrap
#Create RGB colors





#INITIALISE SOME MODULES, SET FUNCTIONS THAT CAN QUERY DATABASE

#Import modules



#Send postgresSQL command
def arbitrary_command_newdatabase(command):
    #Connect-to-database info
    user_pargo = "pargo_employee"
    host_pargo = "31.3.103.66"
    password_pargo = "146J4XY8dQYqZrVcUhaY"
    database_pargo = "pargo"
    port_pargo = "9364"
    
    #Connect to database 
    connection = psycopg2.connect(user= user_pargo,
                          password= password_pargo,
                          host= host_pargo,
                          port= port_pargo,
                          database= database_pargo)
    cursor = connection.cursor()
    #Try to send commadn. If not, print the SQL error
    try:
        cursor.execute(str(command))
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        return results, column_names
    except psycopg2.DatabaseError as e:

        # Message stating export unsuccessful.
        print("Data export unsuccessful.")
        traceback.print_exc()

    except TypeError as e:
        print("TypeError occured")
        print(e)
    finally:
        # Close database connection.
        connection.close()
        

#Send SQL command
def arbitrary_command_nocols(command):
    #Connect-to-database info
    user_pargo = "pargo_employee"
    host_pargo = "31.3.103.66"
    password_pargo = "146J4XY8dQYqZrVcUhaY"
    database_pargo = "pargo"
    port_pargo = "9364"

    connection = psycopg2.connect(user= user_pargo,
                          password= password_pargo,
                          host= host_pargo,
                          port= port_pargo,
                          database= database_pargo)
    cursor = connection.cursor()
    try:
        cursor.execute(str(command))
        results = cursor.fetchall()
        return results
    except psycopg2.DatabaseError as e:

        # Message stating export unsuccessful.
        print("Data export unsuccessful.")
        traceback.print_exc()

    except TypeError as e:
        print("TypeError occured")
        traceback.print_exc()
    finally:
        # Close database connection.
        connection.close()
        
def arbitrary_command_noresults(command): #Useful for debugging
    #Connect-to-database info
    user_pargo = "pargo_employee"
    host_pargo = "31.3.103.66"
    password_pargo = "146J4XY8dQYqZrVcUhaY"
    database_pargo = "pargo"
    port_pargo = "9364"
    
    

    connection = psycopg2.connect(user= user_pargo,
                          password= password_pargo,
                          host= host_pargo,
                          port= port_pargo,
                          database= database_pargo)
    cursor = connection.cursor()
    try:
        cursor.execute(str(command))
        #results = cursor.fetchall()
        #return results
    except psycopg2.DatabaseError as e:
    
        # Message stating export unsuccessful.
        print("Data export unsuccessful.")
        traceback.print_exc()
    except TypeError as e:
        print("TypeError occured")
        print(e)
    finally:
        # Close database connection.
        connection.close()
        
        
def send_command_to_mysql_database(command):
    """ Connect to MySQL database"""
    conn = None
    user_pargo = "keesvanbezouw"
    host_pargo = "185.21.189.94"
    password_pargo = "ucMGmjKyMz6XJUA9"
    database_pargo = "pargo_db"
    port_pargo = "9364"
    try:
        conn = mysql.connector.connect(host= host_pargo,
                                       database= database_pargo,
                                       user= user_pargo,
                                       password=password_pargo)
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute(str(command))

            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return results, column_names
        
 
    except Error as e:
        print(e)
 
    finally:
        cursor.close()
        conn.close()
 
    return results, column_names


#Timestamps of 'w2p is in {confirmed, ready_for_delivery, and completed are collected from order_processing}. Timestamp for
# at collection point is gained from courier_tracking_info
def extract_timestamps_from_sql(company, month_nr, year_nr):
    #FIRST SELECT CONFIRMED, AT COURIER FOR DELIV, COMPLETED FROM TRACKING INFO
    command = ("SELECT "
    "order_process_tracking_info.order_process_id AS order_id, "
    "order_process_tracking_info.created_at AS timestamp, "
    "process_types_stages.name AS process_type_stage_name "
    "FROM order_process_tracking_info " 
        "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
        "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
    "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
    "WHERE (process_types_stages.name = 'confirmed' OR process_types_stages.name = 'at_courier_for_delivery' OR process_types_stages.name = 'completed') " 
    "AND order_process_tracking_info.order_process_id IN ( "
        "SELECT order_process_tracking_info.order_process_id "
        "FROM order_process_tracking_info " 
            "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
            "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
            "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
            "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
            "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
        "WHERE process_types.name = 'w2p' "
            "AND process_types_stages.name = 'confirmed' "
            "AND EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
            " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " " 
            "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + "); " )

    try:
        results, column_names = arbitrary_command_newdatabase(command)
        results_order_process_df = pd.DataFrame(results)
        results_order_process_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_order_process_df = pd.DataFrame(columns=['order_id', ' timestamp', 'process_type_stage_name'])

    #EXTRACT SIMILAR INFORMATION BUT NOW INCLUDING THE INFORMATION AS GIVEN BY THE COURIER.
    #THEN SELECT POD AS GIVEN BY THE COURIER
    
#At collection point collected van courier_tracking_info. MAAK DIT LEIDEND

    command = ("SELECT "
    "courier_tracking_info.order_process_id AS order_id, "
    "courier_tracking_info.tracking_datetime AS timestamp, "   
    "courier_tracking_info_stages.name AS courier_type_stage_name "
    "FROM courier_tracking_info "
    "INNER JOIN courier_tracking_info_stages ON courier_tracking_info_stages.uuid = courier_tracking_info.courier_tracking_info_stage_id "
    "WHERE courier_tracking_info_stages.name = 'delivered' "
            #Multiple actions in courier_tracking_info are brought together under 'delivered'
        "AND courier_tracking_info.order_process_id IN ( "
            "SELECT order_process_tracking_info.order_process_id "
            "FROM order_process_tracking_info " 
                "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
                "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
                "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
            # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
                "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
            "WHERE process_types.name = 'w2p' "
                "AND process_types_stages.name = 'confirmed'  "
                "AND EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
                " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
                "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + "); ")

    try:
        results, column_names = arbitrary_command_newdatabase(command)
        results_courier_df = pd.DataFrame(results)
        results_courier_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_courier_df = pd.DataFrame(columns=['order_id', ' timestamp', 'courier_type_stage_name'])

    # MERGE RESULTS
    #When multiple order_ids, make sure to remove multi-indexing
    results_order_process_df.columns = results_order_process_df.columns.get_level_values(0)
    results_courier_df.columns = results_courier_df.columns.get_level_values(0)
  
    return results_courier_df, results_order_process_df 


#If you have the timestamps of the four order process types, each linked with an order_id, now merge them so that the four timestamps
#Are each in one row and labelled properly.
def merge_and_tidy_timestamps(results_courier_df, results_order_process_df):
    #Preprocess to eventually merge all
    df_confirmed_dupl = results_order_process_df[results_order_process_df['process_type_stage_name'] == "confirmed"]

    df_at_courier_for_deliv_dupl = results_order_process_df[results_order_process_df['process_type_stage_name'] == "at_courier_for_delivery"]
    df_at_col_dupl = results_courier_df[results_courier_df['courier_type_stage_name'] == "delivered"]
    df_completed_dupl = results_order_process_df[results_order_process_df['process_type_stage_name'] == "completed"]


    #Prev dataframes
    df_confirmed = remove_duplicates(df_confirmed_dupl, take_max_of = 'timestamp', on = 'order_id')
    df_at_courier_for_deliv = remove_duplicates(df_at_courier_for_deliv_dupl, take_max_of = 'timestamp', on = 'order_id')
    df_at_col = remove_duplicates(df_at_col_dupl, take_max_of = 'timestamp', on = 'order_id')
    df_completed = remove_duplicates(df_completed_dupl, take_max_of = 'timestamp', on = 'order_id')

    df_timestamps_tot = df_confirmed.merge(df_at_courier_for_deliv,on='order_id').merge(df_at_col, on = 'order_id').merge(df_completed, on = 'order_id')

    df_timestamps_tot.columns = ['order_id','status_confirmed','status_at_courier_for_deliv','status_at_collection_point','status_completed']
    return df_timestamps_tot


##########################################################################
def query_timestamps_mysql(company, year_nr, month_nr):
    command = ("SELECT "
    "orders_status.ordersid AS order_id, "
    "orders_status.order_status_open AS status_confirmed, "
    "CASE WHEN orders_status.courier_order_recieved IS NOT NULL "
        "THEN orders_status.courier_order_recieved "
        "ELSE orders_status.order_exported_carrier END "
        "AS status_at_courier_for_deliv, "
    "CASE WHEN orders_status.courier_pod IS NOT NULL "
        "THEN orders_status.courier_pod "
        "ELSE orders_status.order_status_in_stock END "
        "AS status_at_collection_point, "
    "orders_status.order_status_collected AS status_completed "
    "FROM orders_status "
        "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
        "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid  "
    "WHERE orders_status.courier_pod IS NOT null " #This finds all delivered parcels
        "AND orders_status.order_exported_carrier IS NOT null " #Also, some dummy entries are entered in the database.. THis filters these
        "AND orders_status.order_status_open IS NOT NULL  "
         " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
        "AND MONTH (orders_status.order_status_open) = " + str(month_nr) + " "
        "AND YEAR (orders_status.order_status_open) = " + str(year_nr) + " "
    ";")

    try:
        results, column_names = send_command_to_mysql_database(command)
        results_df = pd.DataFrame(results)
        results_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_df = pd.DataFrame(columns=['order_id', 'status_confirmed', 'status_at_courier_for_deliv', 'status_at_collection_point', 'status_completed'])

    # MERGE RESULTS
    #When multiple order_ids, make sure to remove multi-indexing
    results_df.columns = results_df.columns.get_level_values(0)
  
    return results_df

#################################################################
#Only take business days in account when calculating delivery times.
def extract_deliv_times_in_days(df_timestamps_tot):
    #isolate dates of stuff where dates are relevant
    at_courier_date = [df_timestamps_tot['status_at_courier_for_deliv'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]
    at_col_date = [df_timestamps_tot['status_at_collection_point'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]

    at_courier_date = np.asarray(at_courier_date)
    at_col_date = np.asarray(at_col_date)

    deliv_times_days = [calc_daydiff(at_courier_date[i], at_col_date[i], corrected = True) for i in range(at_courier_date.shape[0])]

    deliv_times_np_days = np.asarray(deliv_times_days)
    deliv_times_np_days = deliv_times_np_days[deliv_times_np_days >= 0] #Just to make sure that only correct values are taken
    return deliv_times_np_days

def extract_timediffs_corrected(time_and_date_column_a, time_and_date_column_b, corrected = True):
    dates_column_a = [time_and_date_column_a[i].date() for i in range(len(time_and_date_column_a))]
    dates_column_b = [time_and_date_column_b[i].date() for i in range(len(time_and_date_column_b))]

    dates_a = np.asarray(dates_column_a)
    dates_b = np.asarray(dates_column_b)

    corrected_daydiffs = [calc_daydiff(dates_a[i], dates_b[i], corrected = corrected) for i in range(time_and_date_column_a.shape[0])]
    
    corrected_daydiffs_np = np.asarray(corrected_daydiffs)
    #corrected_daydiffs_np =  corrected_daydiffs[corrected_daydiffs_np >= 0] #Just to make sure that only correct values are taken
    return corrected_daydiffs_np

##################################################################################
#Don't take business days into account when calculating collection times
def extract_collection_times_in_days(df_timestamps_tot):
    #isolate dates of stuff where dates are relevant
    collected_date = [df_timestamps_tot['status_completed'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]
    at_col_date = [df_timestamps_tot['status_at_collection_point'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]

    collected_date = np.asarray(collected_date)
    at_col_date = np.asarray(at_col_date)

    collection_times_days = [calc_daydiff(at_col_date[i], collected_date[i], corrected = False) for i in range(collected_date.shape[0])]

    collection_times_np_days = np.asarray(collection_times_days)
    collection_np_after_x_days =  collection_times_np_days[ collection_times_np_days >= 0] #Just to make sure that only correct values are taken
    return collection_np_after_x_days

#################################################################################################################################
#Exxtract information that's relevant when calculating the days of the month at which people pick up parcles
#And the day of the week (monday, tuesday etc)
def extract_collected_dates_and_weekdays(df_timestamps_tot):
    collected_date =  [df_timestamps_tot['status_completed'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]
    collected_date = np.asarray(collected_date) # creates an array of collected dates
    # started commenting on some code by myself-abu.


    collection_weekdays = [collected_date[i].weekday() for i in range(collected_date.shape[0])] #INDEXING STARTS AT 0 HERE!! (as opposed to 1 in .isoweekday())
    collected_days = [collected_date[i].day for i in range(collected_date.shape[0])]
    weekday_str_to_index = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"} 
    weekdays = [weekday_str_to_index[i] for i in range(len(weekday_str_to_index))]
    return collected_date, collected_days, weekdays, collection_weekdays



#####################################################################
# PLOT SALES PER TIME
def plot_collection_time_of_days(collection_hours, title = ""):
    hours_in_day = [i + 0.5 for i in range(24)] #Add half hours to put the value of a between hour in the middle, for now
    hours_in_day_round = [i for i in range(24)] 
    nr_collection_per_hour = [np.count_nonzero(collection_hours == i) for i in range(24)]
    
    #Only plot hours where people actually pick up parcels
    last_nonzero = [index for index, item in enumerate(nr_collection_per_hour) if item is not 0][-1]
    first_nonzero = [index for index, item in enumerate(nr_collection_per_hour) if item is not 0][0]
   
    begin_hour = hours_in_day_round[first_nonzero] - 1
    end_hour = hours_in_day_round[last_nonzero] + 1
    
    #But always show from atleast 6 til 21
    if begin_hour > 7 :
        begin_hour = 6
    if end_hour < 21:
        begin_hour = 21
    
    fig, ax = plt.subplots(figsize = (5,2))
    plt.plot(hours_in_day[begin_hour:end_hour], nr_collection_per_hour[begin_hour:end_hour])
    plt.xlim(begin_hour, end_hour)
    plt.ylim(0, 1.2*max(nr_collection_per_hour))
    
    #Add accompanying text
    for i in range(len(hours_in_day)):
        if nr_collection_per_hour[i] is not 0:
            plt.text(x = hours_in_day[i] , y = nr_collection_per_hour[i]+ 0.08*max(nr_collection_per_hour), s = nr_collection_per_hour[i], ha = 'center')
     
    # these are the plots, sets the Title and x/y labels
    plt.grid(False)
    plt.title(title)
    plt.xlabel("Time of the day")
    plt.ylabel("Number of collections")
    
    
    xtick_str = [ str(int(np.linspace(begin_hour, end_hour, end_hour - begin_hour + 2 )[i]))  + ":00" for i in range(0, len(np.linspace(begin_hour, end_hour, end_hour - begin_hour + 1))) ]
    
    xtick_str[0] = "Before " + str(begin_hour) + ":00"
    xtick_str[-1] = "After " + str(end_hour + 1) + ":00"
    
    
    plt.xticks(hours_in_day_round[begin_hour: end_hour], xtick_str, rotation = 45, ha = 'right')
    plt.show()
    return fig
    

#Extract all parcels/order IDs + the store that they were delivered to.
def deliveries_per_pup_sql(company, month_nr, year_nr, show_top_x = 10):
    command_new_db = ("SELECT order_process_tracking_info.uuid AS uuid, " 
    # "process_types.name AS process_type_name, "
    "order_process_tracking_info.order_process_id AS order_id, "
    #"suppliers.name AS sender_name, "
    "pickup_points.name AS stores "
    "FROM order_process_tracking_info "
    "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id "
    "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
    "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
    "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
    "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "
    "WHERE "
        " EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
        " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
        
        "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + " "
        "AND process_types_stages.name = 'at_collection_point' "
        "AND process_types.name = 'w2p'; ")

    try:
        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        results_new_db_df = pd.DataFrame(results_new_db)
        results_new_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_new_db_df = pd.DataFrame(columns=['uuid', 'order_id', 'stores'])


    command_old_db = ("SELECT "
            "orders_status.ordersid AS uuid, "
            "orders.ordersid AS order_id, "
            "pargopoint.store_name AS stores "
        "FROM orders_status "
            "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
            "INNER JOIN pargopoint ON pargopoint.pargopointid = orders.pargopointid "
            "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
        "WHERE "
            " MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
             " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
            "AND YEAR(orders_status.order_status_open) = " + str(year_nr) + " "
            "AND orders_status.order_status_collected IS NOT NULL; ")

    try:
        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        results_old_db_df = pd.DataFrame(results_old_db)
        results_old_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_old_db_df = pd.DataFrame(columns=['uuid', 'order_id', 'stores'])

    results_old_db_df.columns = results_old_db_df.columns.get_level_values(0)
    results_new_db_df.columns = results_new_db_df.columns.get_level_values(0)

    results_df = pd.concat([results_old_db_df, results_new_db_df], sort = False) 

    #Find all unique store values and their counts aka number of parcels per store.
    values, counts = np.unique(results_df['stores'], return_counts=True)
    dummy_dict = { 'STORES': values,'VOLUME': counts} #Easiest way to convert to lists to dataframe type is via a dictionary
    dummy_table = pd.DataFrame(dummy_dict)


    #Make sure that no indexing error occurs when the number of popups happen to be less than show_top_x
    if show_top_x >= len(values):
        show_top_x = len(values)
    dat_top = pd.DataFrame({'NUMBER': [i for i in range(1,show_top_x + 1)]})
    table = dummy_table.sort_values(by=['VOLUME'], ascending=False) 
    table_merged = dat_top.join(table[0:show_top_x].reset_index(drop = True)) #After sorting, indices are distorted. Reset indices so that joining works 
    #labels = ("Top", "Stores", "Volume")
    #values = [[i for i in range(1,11)], table["Stores"][0:10].values.tolist(), table["Volume"][0:10].values.tolist()]
    #values = list(map(list, zip(*values)))
    return table_merged

def calc_daydiff(date_a, date_b, corrected = True): #calcs date_b - date_a
    day_diff= 0
    date_a0 = date_a
    date_b0 = date_b
    delta = datetime.timedelta(1)
    holidays_SA_2018 = [] #add this later 
    holidays_SA_2019 = [datetime.datetime(2019, 1, 1), datetime.datetime(2019, 3, 21), datetime.datetime(2019, 4, 19), datetime.datetime(2019, 4, 22), datetime.datetime(2019, 4, 27), datetime.datetime(2019, 5, 1), datetime.datetime(2019, 6, 17), datetime.datetime(2019, 9, 24), datetime.datetime(2019, 12, 16), datetime.datetime(2019, 12, 25), datetime.datetime(2019, 12, 26)]
    try:
        if((date_b - date_a).days > 0 ): #If b is larger than a
            while date_a != date_b:
                #subtract a day
                date_b -= delta
                if corrected == True:
                    if (date_b.isoweekday() not in (6, 7) and date_b not in holidays_SA_2019): #indices 6,7 stand for sat, sun. Include holidays later
                        day_diff += 1
                else:
                    day_diff += 1
        else: 
            while date_a != date_b:
                #subtract a day
                date_b += delta

                # if not saturday or sunday, add to count
                if corrected == True:
                    if (date_b.isoweekday() not in (6, 7) and date_b not in holidays_SA_2019): #indices 6,7 stand for sat, sun. Include holidays later
                        day_diff -= 1
                else:
                    day_diff -= 1
        return day_diff
    except AttributeError as E:
        print(E)
        day_diff =  np.datetime64('NaT')
    except OverflowError as E: 
        print(E)
        print(date_a0)
        print(date_b0)
        
#################################################
def find_collected_not_collected_prev_x_months(company, year_nr, month_nr, nr_months_back = 6):
    ym_array, ym_array_str =  make_year_month_array(nr_months_back = nr_months_back)
    results_df = pd.DataFrame(columns = ['ym_stamp', 'collected', 'not_collected', 'collected_pct'])
    for i, year_month in enumerate(ym_array):

        command_old_db = ("SELECT "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_collected' THEN 1 ELSE 0 END) AS status_collected, "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_return_route' THEN 1 ELSE 0 END) AS status_return_route, "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_return_supplier' THEN 1 ELSE 0 END) AS status_return_supplier "
                "FROM orders_status "
                    "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
                    "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
                    "INNER JOIN settings_status ON settings_status.settings_statusid = orders.statusid "
                "WHERE "
                    "MONTH(orders_status.order_status_open) = " + str(year_month[1]) + " "
                    " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "

                    "AND YEAR(orders_status.order_status_open) = " + str(year_month[0]) + " " 
                ";  ")

        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        parcel_counts_old_db_df = pd.DataFrame(results_old_db)
        parcel_counts_old_db_df.columns = [column_names]
        parcel_counts_old_db_df.columns = parcel_counts_old_db_df.columns.get_level_values(0)
        parcel_counts_old_db_df = parcel_counts_old_db_df.fillna(0)

        parcels_collected_old = parcel_counts_old_db_df["status_collected"][0] 
        parcels_not_collected_old = parcel_counts_old_db_df["status_return_route"][0] + parcel_counts_old_db_df["status_return_supplier"][0]

        #Now find for new database
        command_new_db = ("SELECT "
                    "SUM(CASE WHEN process_types_stages.name = 'completed' THEN 1 ELSE 0 END) AS status_collected, "
                    "SUM(CASE WHEN process_types_stages.name = 'not_completed' THEN 1 ELSE 0 END) AS status_not_collected "
                    "FROM order_processes "
                        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_processes.current_process_stage "
                        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                        "INNER JOIN process_types ON process_types.uuid = order_processes.process_type_id "
                    "WHERE "
                         "  EXTRACT (MONTH from order_processes.created_at) = " + str(year_month[1]) + " "
                         " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
                        "AND EXTRACT (YEAR from order_processes.created_at) = " + str(year_month[0]) + " "
                        "AND process_types.name  = 'w2p' ") #Multiple process type stages from different process types (eg p2p etc) have teh same name

        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        parcel_counts_new_db_df = pd.DataFrame(results_new_db)
        parcel_counts_new_db_df.columns = [column_names]
        parcel_counts_new_db_df.columns = parcel_counts_new_db_df.columns.get_level_values(0)
        parcel_counts_new_db_df = parcel_counts_new_db_df.fillna(0)

        parcels_collected_new = parcel_counts_new_db_df["status_collected"][0] 
        parcels_not_collected_new = parcel_counts_new_db_df["status_not_collected"][0]

        nr_parcels_collected = parcels_collected_old + parcels_collected_new
        nr_parcels_not_collected =  parcels_not_collected_old + parcels_not_collected_new
        if nr_parcels_collected + nr_parcels_not_collected !=0: #Avoid division by 0
            frac = nr_parcels_collected/(nr_parcels_collected + nr_parcels_not_collected)
        else:
            frac = 0
        results_df = results_df.append({'ym_stamp': (str(ym_array_str[i][1]) + " " + str(ym_array_str[i][0])), 'collected' : nr_parcels_collected, 'not_collected': nr_parcels_not_collected, 'collected_pct': 100 * frac} , ignore_index=True)

    if [index for index, item in enumerate(results_df['collected']) if item == 0]:
        last_zero_element = [index for index, item in enumerate(results_df['collected']) if item == 0][-1]
        results_df_nonzero = results_df.iloc[last_zero_element + 1:]
        ym_array_str_nonzero = ym_array_str[last_zero_element + 1:]
        return results_df_nonzero, ym_array_str_nonzero
    else: 
        return results_df, ym_array_str


        

def increment_months(orig_date, nr_months):
    # advance year and month by one month
    new_year = orig_date.year
    new_month = orig_date.month + nr_months
    # note: in datetime.date, months go from 1 to 12
    if new_month > 12 and new_month <= 24:
        new_year += 1
        new_month -= 12
    
    if  new_month > 24:
        new_year += 2
        new_month -= 24
        
    if new_month < 1 and new_month >= -11:
        new_year -= 1
        new_month += 12
        
    if new_month < -11:
        new_year -= 2
        new_month += 24

    last_day_of_month = calendar.monthrange(new_year, new_month)[1]
    new_day = min(orig_date.day, last_day_of_month)

    return orig_date.replace(year=new_year, month=new_month, day=new_day)

#Sometimes there's multiple timestamps for the same process type stage name. Remove those
def remove_duplicates(df_w, take_max_of, on =  'order_id'):
    #Add this line because of: https://stackoverflow.com/questions/43298192/valueerror-grouper-for-something-not-1-dimensional
    df_w.columns = df_w.columns.get_level_values(0)
    df_wo = df_w.groupby([str(on)]).min()[str(take_max_of)].reset_index()
    df_wo.columns = df_wo.columns.get_level_values(0)
    return df_wo




def make_h_bar_stacked(labels, segments, dat,  xlabel = 'Number of parcels', ylabel = "Days", title = " "): #Add percentages later
    #https://stackoverflow.com/questions/21397549/stack-bar-plot-in-matplotlib-and-add-label-to-each-section-and-suggestions
    # generate some multi-dimensional data & arbitrary labels
    data = np.asarray(dat)
    #What percentages does pargo want here?
    #percentages = (np.random.randint(5,20, (len(labels), segments)))
    y_pos = np.arange(len(labels))

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    #Official pargo colours
    colors = np.asarray([[255,242,0], [24,120,185], [223, 224, 224], [0,0,0]])/255 
    patch_handles = []
    left = np.zeros(len(labels)) # left alignment of data starts at zero
    for i,d in enumerate(data):
        patch_handles.append(ax.barh(y_pos, d, 
          color=colors[i%len(colors)], align='center', 
          left=left))
        # accumulate the left-hand offsets
        left = left + d

    # go through all of the bar segments and annotate
    for j in range(len(patch_handles)):
        for i, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            #ax.text(x,y, "%d%%" % (percentages[i,j]), ha='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    ax.grid(False)
    plt.title(str(title))
    plt.show()
    return ax.get_figure()


def make_h_bar(labels, segments, dat, fig_size = (10,8), xlabel = 'Percentage of parcels', ylabel = "Days", title = " ", color_id = 0):
    """
     Make histogram graphs.
    """
    
    data = np.asarray(dat)
    #What percentages does pargo want here?
    percentages = [data[i]/sum(data)*100 for i in range(len(data))]
    percentages_np = np.asarray(percentages)
    y_pos = np.arange(len(day_labels))

    fig = plt.figure(figsize= fig_size)
    ax = fig.add_subplot(111)
    #Official pargo colours
    
    colors = np.asarray([[255,242,0], [24,120,185], [223, 224, 224], [0,0,0]])/255 
    patch_handles = []
    left = np.zeros(len(day_labels)) # left alignment of data starts at zero

    patch_handles.append(ax.barh(y_pos, percentages_np, 
    color=colors[color_id%len(colors)], align='center', 
    left=left))
    # accumulate the left-hand offsets

    # go through all of the bar segments and annotate
    for j in range(len(patch_handles)):
        for i, patch in enumerate(patch_handles[j].get_children()):
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            if int(percentages[i]) != 0:
                ax.text(x,y, "%d%%" % (percentages[i]), ha='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    max_rounded = (math.ceil(max(percentages_np) / 10.0)) * 10
    ax.set_xticks(np.linspace(0,max_rounded, int(max_rounded/10 + 1)) ) # floating issue
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel(str(ylabel))
    ax.grid(False) 
    plt.title(str(title))
    plt.tight_layout()
    plt.show()
    return ax.get_figure()

#Extract some parameters that are useful later on
def extract_parameters(deliv_times_np_days, df_timestamps_tot, stack_from_x_days = 6):
    unique_days, unique_days_counts = np.unique(deliv_times_np_days,  return_counts=True)


    nr_of_delivered_parcels = df_timestamps_tot.shape[0]
    stack_from_x_days = 6


    for i in range(0, stack_from_x_days + 1):
        if i not in unique_days:
            unique_days_counts = np.insert(unique_days_counts, i , 0)

    #Make a linspace with the number of days until the days that you stack.
    days_till_stack = np.linspace(0, stack_from_x_days - 1 , stack_from_x_days)
    #x_bar is the format to go into an hbar (actually y-axis but x is the indep. variable)
    x_bar = np.append(days_till_stack, stack_from_x_days) #from >7 days, stack the results. 8 days means 8+ days. Take note in the x-ticks

    #Dependent variable info
    y_bar = unique_days_counts[0:stack_from_x_days]
    y_bar = np.append(y_bar, np.sum(unique_days_counts[stack_from_x_days:]))
    day_labels = [str(int(x_bar[i])) for i in range(len(x_bar))]
    day_labels[stack_from_x_days] = str(stack_from_x_days) + "+"
    nr_of_suppliers = 1 #Number of suppliers

    return nr_of_delivered_parcels, day_labels, nr_of_suppliers, unique_days, unique_days_counts, x_bar, y_bar





##############################################################
#Find hours, dates and times when people pick up parcel.
def extract_col_times_hours_dates(df_times):
    collection_values = df_times.values
    collection_times = pd.to_datetime(collection_values) #Life is better in datetime type than in npdatetime
    collection_hours = collection_times.hour + 2 #Correct for difference of time in SA from UCT  
    collection_dates = collection_times.day
    return collection_hours, collection_dates, collection_times

def piechart_collection_after_pre_work(collection_hours, title = " "):
    """
    This function just plots the whether or not parcels have been collected within a particular time.
    """
    collection_after_work = np.count_nonzero(collection_hours >= 17) + np.count_nonzero(collection_hours < 9)
    collection_pre_work = len(collection_hours) - collection_after_work
    assert collection_after_work + collection_pre_work == len(collection_hours)
    fig = pie_chart(title = title, data = [collection_pre_work, collection_after_work], labels = ["Within office hours", "Outside of office hours"])
  
    return fig 
    
def plot_collected_parcels_per_day_of_month(collection_dates, year_nr, month_nr):
    unique_day_of_col, unique_day_of_col_counts = np.unique(collection_dates,  return_counts=True)
    all_days = np.linspace(1, calendar.monthrange(int(year_nr), int(month_nr))[1] , calendar.monthrange(int(year_nr), int(month_nr))[1]) #monthrange returns nr of days in a month. Add 1 to correct for indexing
    #Make some ticks for each individual day later

    #Make sure that days with zero sales are still included
    for i in range(1,  calendar.monthrange(int(year_nr), int(month_nr))[1]  + 1):
        if i not in unique_day_of_col:
            unique_day_of_col_counts = np.insert(unique_day_of_col_counts, i - 1 , 0)
    plt.figure(figsize = (5,2))
    plt.plot(all_days, unique_day_of_col_counts)
    plt.title("Collections per day of the month")
    plt.ylabel("Parcels")
    plt.show()

################################################################################################
def find_nr_of_returned_parcels_sql(company, month_nr, year_nr): #This is by courier
    #FIND NUMBER OF RETURNS PARCEL (#Collected, returned, in stock)
    #Lionel: From reporting perspective, what constitutes a returned parcel
    #When both p2w and status has been on at couier

    #Changing the nested join condition to '"AND order_processes.order_id IN ( "SELECT order_processes.order_id "' solved everysings
    command = ("SELECT COUNT(order_process_tracking_info.uuid) AS nr_of_returned_parcels "
    "FROM order_process_tracking_info " 
        "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
        "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "
    "WHERE process_types.name = 'w2p_not_collected' "
    "AND process_types_stages.name = 'completed' "
    "AND order_processes.order_id IN ( "
        "SELECT order_processes.order_id "
        "FROM order_process_tracking_info " 
            "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
            "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
            "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
            "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
        # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
            "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
            "WHERE process_types.name = 'w2p' "
                "AND process_types_stages.name = 'confirmed' "
                "AND EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
                " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
                "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + "); ")


    results = arbitrary_command_nocols(command)

    results_df = pd.DataFrame(results)
    nr_of_returned_parcels = results_df[0][0]
    return nr_of_returned_parcels

# PIE CHART
def pie_chart_parcel_status(nr_parcels_collected, nr_parcels_returned, nr_parcels_in_stock, title = ""):
    fig = pie_chart(title = title, data = [nr_parcels_collected, nr_parcels_returned, nr_parcels_in_stock], labels = ["Collected", "Not collected", "In stock"])
    return fig


def extract_historical_delivery_rates(company, month_nr, year_nr, nr_months_back = 6):
    year_month_array, year_month_array_str = make_year_month_array(nr_months_back)
    str_middle = ""
    str_timefilter = (" BETWEEN '" + str(int(year_month_array[0][0])) + "-" + str(int(year_month_array[0][1])) + "-01 00:00:01'"
    " AND NOW()")

    command_new_database_courier_info = ("SELECT "
        "courier_tracking_info.order_process_id AS order_id, "
        "courier_tracking_info.tracking_datetime AS timestamp, "   
        "courier_tracking_info_stages.name AS stage_name, "
        "pickup_points.location_type AS location_type "
        "FROM courier_tracking_info "
            "INNER JOIN courier_tracking_info_stages ON courier_tracking_info_stages.uuid = courier_tracking_info.courier_tracking_info_stage_id "                            
            "INNER JOIN order_processes ON order_processes.uuid = courier_tracking_info.order_process_id " 
            "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "    
            "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id " 
        "WHERE courier_tracking_info_stages.name = 'collected' OR courier_tracking_info_stages.name = 'delivered' "
            " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
            "AND courier_tracking_info.tracking_datetime " + str(str_timefilter) + " "";" )


    try:
        results, column_names = arbitrary_command_newdatabase(command_new_database_courier_info)
        timestamps_new_df = pd.DataFrame(results)
        timestamps_new_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        timestamps_new_df = pd.DataFrame(columns=['order_id', ' timestamp', 'stage_name'])

    # MERGE RESULTS
    #When multiple order_ids, make sure to remove multi-indexing
    timestamps_new_df.columns = timestamps_new_df.columns.get_level_values(0)

    if timestamps_new_df.empty == True: 
        command_new_database = ("SELECT "
            "order_process_tracking_info.order_process_id AS order_id, "
            "order_process_tracking_info.created_at AS timestamp, "
            "process_types_stages.name AS stage_name, "
            "pickup_points.location_type AS location_type "
            "FROM order_process_tracking_info " 
                "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
                "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
                "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
            # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
                "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id " 
            "WHERE (process_types_stages.name = 'at_courier_for_delivery' OR process_types_stages.name = 'at_collection_point') " 
                "AND process_types.name = 'w2p' "
                " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " " 
                "AND order_process_tracking_info.created_at " + str(str_timefilter) + " ;")

        try:
            results, column_names = arbitrary_command_newdatabase(command_new_database)
            timestamps_new_df = pd.DataFrame(results)
            timestamps_new_df.columns = [column_names]

        except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
            timestamps_new_df = pd.DataFrame(columns=['order_id', ' timestamp', 'stage_name'])

    timestamps_new_df.columns = timestamps_new_df.columns.get_level_values(0)

    #Make terms homogeneous to extract timediffs easier
    timestamps_new_df = timestamps_new_df.replace("collected", "at_courier_for_delivery")
    timestamps_new_df = timestamps_new_df.replace("delivered", "at_collection_point")
    timestamps_merged_df = timestamps_new_df[timestamps_new_df["stage_name"] == "at_courier_for_delivery"].merge(timestamps_new_df[timestamps_new_df["stage_name"] == "at_collection_point"], on = "order_id")
    timestamps_merged_df.columns = ['order_id','timestamp_at_courier', 'stage_name_at_courier','loc_type','timestamp_at_col', 'stage_name_at_col', 'loc_type_double']
    timestamps_tidied_new_df = timestamps_merged_df[["timestamp_at_courier", "timestamp_at_col", "loc_type"]]

    command_old_db = ("SELECT "
        "CASE WHEN orders_status.courier_order_recieved IS NOT NULL "
            "THEN orders_status.courier_order_recieved "
            "ELSE orders_status.order_exported_carrier END "
            "AS timestamp_at_courier, "
        "CASE WHEN orders_status.courier_pod IS NOT NULL "
            "THEN orders_status.courier_pod "
            "ELSE orders_status.order_status_in_stock END "
            "AS timestamp_at_col, "
        "pargopoint.location_type AS loc_type "
        "FROM orders_status "
            "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
            "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
            "INNER JOIN pargopoint ON pargopoint.pargopointid = orders.pargopointid "
        "WHERE orders_status.courier_pod IS NOT null " #This finds all delivered parcels
            "AND orders_status.order_exported_carrier IS NOT null " #Also, some dummy entries are entered in the database.. THis filters these
            "AND orders_status.order_status_open IS NOT NULL  "
             " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
             "AND orders_status.order_status_open " + str(str_timefilter) + " "
        ";")

    #try:
       # results, column_names = send_command_to_mysql_database(command_old_db)
       # timestamps_tidied_old_df = pd.DataFrame(results)
      #  timestamps_tidied_old_df.columns = [column_names]
    #except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
     #   timestamps_tidied_old_df = pd.DataFrame(columns=["timestamp_at_courier", "timestamp_at_col", "loc_type"])
    #timestamps_tidied_old_df.columns = timestamps_tidied_old_df.columns.get_level_values(0)

    timestamps_tidied_df = timestamps_tidied_new_df.copy()
    timestamps_tidied_df.columns = timestamps_tidied_df.columns.get_level_values(0)
    timestamps_tidied_main = timestamps_tidied_df[timestamps_tidied_df["loc_type"] == "main"].reset_index(drop = True)
    timestamps_tidied_reg = timestamps_tidied_df[timestamps_tidied_df["loc_type"] == "regional"].reset_index(drop = True)

    data_df = pd.DataFrame(columns = ["ym", "tot_within_sla", "tot_outside_sla", "percentage_sla"])
    for i in range(len(year_month_array)):
        time_filtered_timestamps_main = timestamps_tidied_main[((timestamps_tidied_main["timestamp_at_courier"].dt.month == int(year_month_array[i][1])) & (timestamps_tidied_main["timestamp_at_courier"].dt.year == int(year_month_array[i][0]))) ] 
        time_filtered_timestamps_reg = timestamps_tidied_reg[((timestamps_tidied_reg["timestamp_at_courier"].dt.month == int(year_month_array[i][1])) & (timestamps_tidied_reg["timestamp_at_courier"].dt.year == int(year_month_array[i][0]))) ] 
        timediffs_main = extract_timediffs_corrected(time_filtered_timestamps_main["timestamp_at_courier"].reset_index(drop = True),  time_filtered_timestamps_main["timestamp_at_col"].reset_index(drop = True), corrected = True)
        timediffs_reg = extract_timediffs_corrected(time_filtered_timestamps_reg["timestamp_at_courier"].reset_index(drop = True), time_filtered_timestamps_reg["timestamp_at_col"].reset_index(drop = True), corrected = True)

        main_within_sla = (timediffs_main <= 4).sum()
        regional_within_sla = (timediffs_reg <= 6).sum()
        tot_within_sla = main_within_sla + regional_within_sla
        tot_outside_sla = len(timediffs_main) + len(timediffs_reg) - tot_within_sla
        if tot_within_sla + tot_outside_sla != 0:
            frac = tot_within_sla  /(tot_within_sla + tot_outside_sla)
        else:
            frac = 0

        data_df = data_df.append({"ym": (str(int(year_month_array[i][1]))) + "_" +  str(int(year_month_array[i][0])), "tot_within_sla": tot_within_sla, "tot_outside_sla": tot_outside_sla, "percentage_sla" : 100 * frac}, ignore_index = True )

    if [index for index, item in enumerate(data_df['percentage_sla']) if item == 0]:
        last_zero_element = [index for index, item in enumerate(data_df['percentage_sla']) if item == 0][-1]
        data_df_nonzero = data_df.iloc[last_zero_element + 1:]
        year_month_array_str = year_month_array_str[last_zero_element + 1:]
        return data_df_nonzero, year_month_array_str
    else: 
        return data_df, year_month_array_str




def extract_hist_col_rates(nr_months_back = 6):
    year_month_array, year_month_array_str = make_year_month_array(nr_months_back = nr_months_back)
    hist_col_rates_df = pd.DataFrame(columns = ["ym", "collected", "returned", "collected_perc"])
    for counter, i in enumerate(year_month_array):
        month_nr = year_month_array[counter][1]
        year_nr = year_month_array[counter][0]
        command_old_db = ("SELECT "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_collected' THEN 1 ELSE 0 END) AS status_collected, "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_return_route' THEN 1 ELSE 0 END) AS status_return_route, "
                "SUM(CASE WHEN settings_status.status_name = 'order_status_return_supplier' THEN 1 ELSE 0 END) AS status_return_supplier "
                "FROM orders_status "
                    "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
                    "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
                    "INNER JOIN settings_status ON settings_status.settings_statusid = orders.statusid "
                "WHERE "
                    "MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
                     " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
                    "AND YEAR(orders_status.order_status_open) = " + str(year_nr) + " " 
                ";  ")

        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        parcel_counts_old_db_df = pd.DataFrame(results_old_db)
        parcel_counts_old_db_df.columns = [column_names]
        parcel_counts_old_db_df.columns = parcel_counts_old_db_df.columns.get_level_values(0)
        parcel_counts_old_db_df = parcel_counts_old_db_df.fillna(0)

        parcels_collected_old = parcel_counts_old_db_df["status_collected"][0] 
        parcels_returned_old = parcel_counts_old_db_df["status_return_route"][0] + parcel_counts_old_db_df["status_return_supplier"][0]

        #Now find for new database
        command_new_db = ("SELECT "
                    "SUM(CASE WHEN process_types_stages.name = 'completed' THEN 1 ELSE 0 END) AS status_collected, "
                    "SUM(CASE WHEN process_types_stages.name = 'not_completed' THEN 1 ELSE 0 END) AS status_not_completed "
                    "FROM order_processes "
                        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_processes.current_process_stage "
                        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                        "INNER JOIN process_types ON process_types.uuid = order_processes.process_type_id "
                    "WHERE "
                        " EXTRACT (MONTH from order_processes.created_at) = " + str(month_nr) + " "
                        " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
                        "AND EXTRACT (YEAR from order_processes.created_at) = " + str(year_nr) + " "
                        "AND process_types.name  = 'w2p' ") #Multiple process type stages from different process types (eg p2p etc) have teh same name

        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        parcel_counts_new_db_df = pd.DataFrame(results_new_db)
        parcel_counts_new_db_df.columns = [column_names]
        parcel_counts_new_db_df.columns = parcel_counts_new_db_df.columns.get_level_values(0)
        parcel_counts_new_db_df = parcel_counts_new_db_df.fillna(0)

        parcels_collected_new = parcel_counts_new_db_df["status_collected"][0] 
        parcels_returned_new = parcel_counts_new_db_df["status_not_completed"][0]

        nr_parcels_collected = parcels_collected_old + parcels_collected_new
        nr_parcels_returned = parcels_returned_old + parcels_returned_new

        hist_col_rates_df = hist_col_rates_df.append({"ym": (str(int(year_month_array[counter][1]))) + "_" +  str(int(year_month_array[counter][0])), "collected": nr_parcels_collected, "returned": nr_parcels_returned, "collected_perc": nr_parcels_collected * 100/(nr_parcels_returned + nr_parcels_collected)}, ignore_index = True)

        if [index for index, item in enumerate(hist_col_rates_df.iloc[0,:]) if item == 0]:
            last_zero_element = [index for index, item in enumerate(hist_col_rates_df.iloc[0,:]) if item == 0][-1]
            hist_col_rates_df_nonzero = hist_col_rates_df.iloc[0,last_zero_element + 1:]
            year_month_array_str = year_month_array_str[last_zero_element + 1:]
            return hist_col_rates_df_nonzero, year_month_array_str
        else: 
            return hist_col_rates_df, year_month_array_str

####################################################################################################################################################

print("kernel works")
def pie_chart( data, labels, title = ""):
    #[yellow, blue, grey, black. 
    colors = np.asarray([[255,242,0], [24,120,185], [223, 224, 224], [0,0,0]])/255 
    
    fig = plt.figure(figsize=(4,4))
    plt.pie(data, colors = colors, autopct='%.1f%%')
    plt.title(str(title))
    plt.tight_layout()
    plt.legend(labels, loc= 'center', bbox_to_anchor=(0.5, -0.1))
    return fig 


#### CSAT score
# this charts out the avg CSAT
def pie_circle_chart(title = "", data = None, labels = ["1", "2", "3", "4"]):
    #[yellow, blue, grey, black.
    data = [int(ratings_df['rating_1'].iloc[0]), int(ratings_df['rating_2'].iloc[0]),int(ratings_df['rating_3'].iloc[0]), int(ratings_df['rating_4'].iloc[0])]
    labels_legend = ["1 = Poor", "2 = Neutral", "3 = Good ", "4 = Excellent"]
    #[yellow, blue, grey, black. 
    colors = np.asarray([[0,0,0],[223, 224, 224], [24,120,185],[255,242,0] ])/255 
    

    ## handle the error. If CSAT are filled.  
    try:
        avg = (data[0]*1 + data[1]*2 + data[2]*3 + data[3]*4)/(data[0] + data[1] + data[2] + data[3])
    except ZeroDivisionError:
        avg = 0
        
    fig, ax = plt.subplots(figsize = (4,4))
    plt.pie(data, labels = labels, colors = colors, autopct='%.1f%%')
    plt.title(str(title))
    plt.tight_layout()
    plt.legend(labels_legend, ncol=2, loc= 'center', bbox_to_anchor=(0.5, -0.1))
    ax.text(0, 0, ("Avg. = " + str(round(avg,2))), ha='center', fontsize = 8)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    return fig 
    
    
#HISTORICAL DATA GRAPHS
# Historical data: Number of parcels in previous 12 months. Note that currently it only takes into account the entries in the old
#system

#Create empty numpy array to store months and years of the prev 12 months

def extract_historial_parcel_deliveries_data(company, month_nr, year_nr, nr_months_back = 24):
    year_month_array = np.zeros((nr_months_back, 2)) #Store month number and year here
    year_month_array_str = [] # Create a list with strings that can be used as xtic
    current_date = datetime.datetime(int(year_nr), int(month_nr), 1)
    for counter, decrement in enumerate(range(-nr_months_back + 1, 1)):
        decremented_month = increment_months(current_date, decrement)
        year_month_array[counter][0] = decremented_month.year
        year_str = str(decremented_month.year) #Isj a number so isj good
        year_month_array[counter][1] = decremented_month.month
        month_str = str(calendar.month_abbr[decremented_month.month])                           #convert to three character abbreviatins of month

        year_month_array_str.append([year_str, month_str])

    #QUERY FROM OLD DATABASE
    # Okaye so you column names have to contain at least one letter. Use 'ym_year_month' format now, where year, month are numbers.

    #We have to select a lot. In str_middle select which columns we actually want (sums of number of parcels created in a certain month, year)
    #. End this part with the last month, and note to exclude the comma or SQL crashes.
    #In str_end grab the data that we want to select from.
    str_start = "Select "

    str_middle = ""
    for i in range(0, year_month_array.shape[0] - 1):
        str_middle = (str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END) "
        #"AS " + " jan_2019 "+ ", ")
        "AS ym_" + str(int(year_month_array[i][0])) + "_" + str(int(year_month_array[i][1])) + " , ")

        #str_middle = str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' THEN 1 ELSE 0 END) AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END), "
    str_middle = (str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[-1][0])) + "' AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[-1][1])) + "' THEN 1 ELSE 0 END) "
    "AS ym_" + str(int(year_month_array[-1][0])) + "_" + str(int(year_month_array[-1][1])) + "  ")

    str_end = ("FROM order_process_tracking_info " 
    "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
    "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
    "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
    "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
    # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
    "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
    "WHERE process_types.name = 'w2p' "
        "AND process_types_stages.name = 'confirmed' "                                       
        " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
        "AND order_processes.current_process_stage <> 'f8c666c1-04a3-4e18-ac4e-eee00c572780' " #<> is the IS NOT EQUAL TO operator. The uuid refers to w2p cancelled
    ";" )

    command_new_db = str_start + str_middle + str_end

    results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)

    #results, column_names = arbitrary_command(command)

    #'''
    results_new_db_df = pd.DataFrame(results_new_db)
    results_new_db_df.columns = [column_names]
    
    results_new_db_df.columns = results_new_db_df.columns.get_level_values(0)
    results_new_db_df = results_new_db_df.fillna(0)
    
    #SELECT ALL THE "OPENED" OF A YEAR/MONTH, WHERE TIMESTAMP FINAL IS NOT NULL
    str_start = "Select "

    str_middle = ""
    for i in range(0, year_month_array.shape[0] - 1):
        str_middle = (str_middle + " SUM(CASE WHEN YEAR(orders_status.order_status_open) = '" + str(int(year_month_array[i][0])) + "' AND MONTH(orders_status.order_status_open) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END) "
        #"AS " + " jan_2019 "+ ", ")
        "AS ym_" + str(int(year_month_array[i][0])) + "_" + str(int(year_month_array[i][1])) + " , ")

        #str_middle = str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' THEN 1 ELSE 0 END) AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END), "
    str_middle = (str_middle + " SUM(CASE WHEN YEAR(orders_status.order_status_open ) = '" + str(int(year_month_array[-1][0])) + "' AND MONTH(orders_status.order_status_open) = '" + str(int(year_month_array[-1][1])) + "' THEN 1 ELSE 0 END) "
    "AS ym_" + str(int(year_month_array[-1][0])) + "_" + str(int(year_month_array[-1][1])) + "  ")

    str_end = ("FROM orders_status "
    "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
    "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
    "WHERE orders_status.order_status_open IS NOT NULL "
     " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
    ";")
    command_old_db = str_start + str_middle + str_end
    
    results_old_db, column_names = send_command_to_mysql_database(command_old_db)
    results_old_db_df = pd.DataFrame(results_old_db)
    results_old_db_df.columns = [column_names]
    results_old_db_df.columns = results_old_db_df.columns.get_level_values(0)
    results_old_db_df =results_old_db_df.fillna(0)
    
    results_df = results_old_db_df + results_new_db_df
    
    #Remove months were sales are 0, if any. Looks nicer in the visualisation
    if [index for index, item in enumerate(results_df.iloc[0,:]) if item == 0]:
        last_zero_element = [index for index, item in enumerate(results_df.iloc[0,:]) if item == 0][-1]
        results_df_nonzero = results_df.iloc[0,last_zero_element + 1:]
        year_month_array_str = year_month_array_str[last_zero_element + 1:]
        return results_df_nonzero, year_month_array_str
    else: 
        return results_df, year_month_array_str

def plot_historical_parcel_deliveries_data(historical_data_df, year_month_array_str, title = ""):
    pargo_blue = np.asarray([24,120,185])/255
    x_axis_ym = [(str(year_month_array_str[i][1]) + " " + str(year_month_array_str[i][0])) for i in range(len(year_month_array_str))]
    y_axis_ym = historical_data_df.values.tolist() #For some reason this has size 1 x n, not n x 1, so fix that in next line
    
    try:
        y_axis_ym =  [int(i) for i in y_axis_ym[0]]
    except TypeError:
        y_axis_ym = [int(y_axis_ym[i]) for i in range(len(y_axis_ym))]

    y_pos = np.arange(len(year_month_array_str))
    fig = plt.figure(figsize=(7,2))
    plt.bar(y_pos, y_axis_ym, align='center', alpha=0.5, color = pargo_blue)
    plt.xticks(y_pos, x_axis_ym, rotation = 45, ha = 'right')
    plt.ylabel('Number of deliveries')
    plt.xlabel('Month')
    plt.ylim(0, max(y_axis_ym)*1.2)
    plt.title(title)

    for i in range(len(y_axis_ym)):
        plt.text(x = y_pos[i] , y = y_axis_ym[i]+ 0.05 * max(y_axis_ym), s = y_axis_ym[i], ha = 'center', fontsize = 5)

    plt.show()
    return fig

def find_current_states_parcels(company, year_nr, month_nr):
    command_old_db = ("SELECT "
        "SUM(CASE WHEN settings_status.status_name = 'order_status_collected' THEN 1 ELSE 0 END) AS status_collected, "
        "SUM(CASE WHEN settings_status.status_name = 'order_status_in_stock' THEN 1 ELSE 0 END) AS status_in_stock, "
        "SUM(CASE WHEN settings_status.status_name = 'order_status_in_stock_reminder2' THEN 1 ELSE 0 END) AS status_in_stock_reminder2, "
        "SUM(CASE WHEN settings_status.status_name = 'order_status_in_stock_reminder1' THEN 1 ELSE 0 END) AS status_in_stock_reminder1, "       
        "SUM(CASE WHEN settings_status.status_name = 'order_status_not_collected' THEN 1 ELSE 0 END) AS status_not_collected, "          
        "SUM(CASE WHEN settings_status.status_name = 'order_status_return_route' THEN 1 ELSE 0 END) AS status_return_route, "
        "SUM(CASE WHEN settings_status.status_name = 'order_status_return_supplier' THEN 1 ELSE 0 END) AS status_return_supplier "
        "FROM orders_status "
            "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
            "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
            "INNER JOIN settings_status ON settings_status.settings_statusid = orders.statusid "
        "WHERE "
            "MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
             " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
            "AND YEAR(orders_status.order_status_open) = " + str(year_nr) + " " 
        ";  ")

    results_old_db, column_names = send_command_to_mysql_database(command_old_db)
    parcel_counts_old_db_df = pd.DataFrame(results_old_db)
    parcel_counts_old_db_df.columns = [column_names]
    parcel_counts_old_db_df.columns = parcel_counts_old_db_df.columns.get_level_values(0)
    parcel_counts_old_db_df = parcel_counts_old_db_df.fillna(0)

    parcels_collected_old = parcel_counts_old_db_df["status_collected"][0] 
    parcels_in_stock_old =  parcel_counts_old_db_df["status_not_collected"][0] + parcel_counts_old_db_df["status_in_stock"][0] + parcel_counts_old_db_df["status_in_stock_reminder1"][0]  + parcel_counts_old_db_df["status_in_stock_reminder2"][0] + parcel_counts_old_db_df["status_not_collected"][0]  
    parcels_returned_old = parcel_counts_old_db_df["status_return_route"][0] + parcel_counts_old_db_df["status_return_supplier"][0]

    #Now find for new database
    command_new_db = ("SELECT "
                "SUM(CASE WHEN process_types_stages.name = 'completed' THEN 1 ELSE 0 END) AS status_collected, "
                "SUM(CASE WHEN process_types_stages.name = 'not_completed' THEN 1 ELSE 0 END) AS status_not_completed, "
                "SUM(CASE WHEN process_types_stages.name = 'at_collection_point' THEN 1 ELSE 0 END) AS status_at_collection_point "
                "FROM order_processes "
                    "INNER JOIN process_types_stages ON process_types_stages.uuid = order_processes.current_process_stage "
                    "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                    "INNER JOIN process_types ON process_types.uuid = order_processes.process_type_id "
                "WHERE "
                    " EXTRACT (MONTH from order_processes.created_at) = " + str(month_nr) + " "
                    " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
                    "AND EXTRACT (YEAR from order_processes.created_at) = " + str(year_nr) + " "
                    "AND process_types.name  = 'w2p' ") #Multiple process type stages from different process types (eg p2p etc) have teh same name

    results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
    parcel_counts_new_db_df = pd.DataFrame(results_new_db)
    parcel_counts_new_db_df.columns = [column_names]
    parcel_counts_new_db_df.columns = parcel_counts_new_db_df.columns.get_level_values(0)
    parcel_counts_new_db_df = parcel_counts_new_db_df.fillna(0)


    parcels_collected_new = parcel_counts_new_db_df["status_collected"][0] 
    parcels_in_stock_new =  parcel_counts_new_db_df["status_at_collection_point"][0]  
    parcels_returned_new = parcel_counts_new_db_df["status_not_completed"][0]

    nr_parcels_collected = parcels_collected_old + parcels_collected_new
    nr_parcels_returned = parcels_returned_old + parcels_returned_new
    nr_parcels_in_stock = parcels_in_stock_old + parcels_in_stock_new
    
    return nr_parcels_collected, nr_parcels_returned, nr_parcels_in_stock



def create_data_sheet_pdf(company):
    #save to
    pdf_path = "/home/abulele/Documents/pargo_datasheets/" + str(company) + "_datasheet.pdf"
    c = canvas.Canvas(pdf_path) #Create a pdf object
    pdfmetrics.registerFont(TTFont('Futura', '/home/abulele/Pictures/pargo_fonts/16020_FUTURAM.ttf'))
    #Import Futura Font (https://stackoverflow.com/questions/4899885/how-to-set-any-font-in-reportlab-canvas-in-python)
    c.setFont('')
    c.setFont('Futura', 48, leading = None)
    page_width  = defaultPageSize[0]
    page_height = defaultPageSize[1]
    #c.drawCentredString(page_width / 2.0, 300, "Body text goes in Futura")  #pixel_x, pixel_y
    
    
    c.showPage() #This actually creates the page. Useful for multi-pages

    fig = plt.figure(figsize=(4, 3))
    #plt.plot([1,2,3,4])
    plt.ylabel('some numbers')

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png') #Actually have to stored matplotlib as png intermediate
    imgdata.seek(0)  # rewind the data

    Image = ImageReader(imgdata)
    im_width = 6*inch
    c.drawImage(Image, page_width / 2.0 - im_width/2, page_height / 2.0 - im_width/2, im_width, im_width )

    #Actuarry save
    c.save()

#######################################################################################################
def plot_collection_weekdays(collection_weekdays, title):
    weekday_indices, weekday_counts = np.unique(collection_weekdays,  return_counts=True)
    weekday_str_to_index = weekday_str_to_index = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"} 
    weekdays = np.zeros([len(weekday_str_to_index)])
    weekdays = [weekday_str_to_index[i] for i in weekday_indices]
                            
    y_pos = np.arange(len(weekdays))

    fig, ax = plt.subplots(figsize = (5,2))
    plt.bar(y_pos, weekday_counts, align='center')
    plt.xticks(y_pos, weekdays, rotation = 45, ha = 'right') #angle in degrees
    plt.ylabel('Collections')
    plt.title(title)
    plt.ylim(0, 1.2 *max(weekday_counts))
    plt.grid(False)
    for i in range(len(weekday_counts)):
        plt.text(x = y_pos[i]-0.2 , y = weekday_counts[i]+ 0.05*max(weekday_counts), s = weekday_counts[i])
    ax.set_facecolor = 'w'
    plt.show()
    return fig

####################################################################################################################
def plot_collection_monthdays(collection_monthdays, title):
    days, day_counts = np.unique(collection_monthdays,  return_counts=True)
    
    month_days = np.arange(1, max(collection_monthdays) + 1)
    
    #Insert zeros if days are lacking
    for i in range(1,  calendar.monthrange(int(year_nr), int(month_nr))[1]  + 1):
        if i not in days:
            day_counts = np.insert(day_counts, i - 1 , 0)


    y_pos = np.arange(1, calendar.monthrange(int(year_nr), int(month_nr))[1] +1)

    fig, ax = plt.subplots(figsize=(5,1.7)) #width, height
    plt.bar(y_pos, day_counts, align='center')
    plt.xticks(y_pos, month_days, rotation = 45) #angle in degrees
    plt.ylabel('Number of collections')
    plt.title(title)
    plt.xlabel('Day of the month')
    plt.ylim(0, 1.1 *max(day_counts))
    plt.grid(False)
    for i in range(len(day_counts)):
        plt.text(x = y_pos[i] , y = day_counts[i]+ 0.05*max(day_counts), s = day_counts[i], ha = 'center')
    ax.set_facecolor = 'w'
    plt.show()

    return fig


#########################################################################################################################################################
def plot_collected_not_collected_df(collected_not_collected_df, title = ""):
    
    colors = np.asarray([[255,242,0], [24,120,185], [223, 224, 224], [0,0,0]])/255 
    y_pos = np.arange(len(collected_not_collected_df))
    y_min = 80
    fig, ax = plt.subplots(figsize=(5,2)) #width, height
    plt.bar(y_pos, collected_not_collected_df["collected_pct"] - y_min, align='center', color = colors[0])
    plt.xticks(y_pos, collected_not_collected_df["ym_stamp"], rotation = 45, ha = 'right') #angle in degrees
    plt.ylabel('Collection rate')
    plt.title(title)
    plt.xlabel('Month')

    ax.set_ylim([0, 23])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["80", "85", "90", "95", "100"])
    plt.grid(False)
    for i in range(len(collected_not_collected_df)):
        plt.text(x = y_pos[i] , y = float(collected_not_collected_df["collected_pct"].iloc[i]) - y_min + 0.01*float(max(collected_not_collected_df["collected_pct"])), s = str("%.2f" % float(collected_not_collected_df["collected_pct"].iloc[i])), ha = 'center')
    ax.set_facecolor = 'w'
    plt.show()
    return fig


def scale_image(drawing, scaling_factor): #Use foo bar so that scaling happens only once!
    """
    Scale a reportlab.graphics.shapes.Drawing()
    object while maintaining the aspect ratio
    """
    scaling_x = scaling_factor
    scaling_y = scaling_factor
    foo_bar = drawing
 
    foo_bar.width = foo_bar.minWidth() * scaling_x
    foo_bar.height = foo_bar.height * scaling_y
    foo_bar.scale(scaling_x, scaling_y)
    return foo_bar

def scale_image_to_width(drawing, to_width): #Use foo bar so that scaling happens only once!
    """
    Scale a reportlab.graphics.shapes.Drawing()
    object while setting the width specifically
    """

    foo_bar = drawing
    aspect_ratio = foo_bar.height/foo_bar.width
    foo_bar.width = to_width
    foo_bar.height = to_width * aspect_ratio
    return foo_bar


 
def from_fig_to_png(fig):
    #Save matplotlib as a png so that it can be exported to pdf after   
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png') #Actually have to stored matplotlib as png intermediate. dpi = x to improve quality
    
    imgdata.seek(0)  # rewind the data

    Image = ImageReader(imgdata)
    return Image

def from_fig_to_svg(fig):
    #Save matplotlib as a svg so that it can be exported to pdf after  
    path = "/home/abulele/Pictures/test.svg"
    if os.path.isfile(path):
        os.remove(path)   # Opt.: os.system("rm "+strFile)
        print("removed file")
    fig.savefig(path, bbox_inches = 'tight',pad_inches = 0, facecolor = 'w', edgecolor='none')
    svg_im = svg2rlg(path)

    return svg_im


def create_table(labels, data):
    fig, axs =plt.subplots(1,1)

    axs.axis('tight')
    axs.axis('off')
    the_table = axs.table(cellText=data,colLabels=labels,loc='center')
    
    
    
    return the_table

def render_mpl_table(data, col_widths = [2,8,2], row_height=0.625, font_size=6,
                     header_color= np.asarray([24,120,185])/255, row_colors=[[223/255, 224/255, 224/255], 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, scale_row = 1, fig_height = None, figwidth = 10,
                     ax=None, **kwargs):
    if ax is None:
        #size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(1,1)
        #ax.axis('tight')
        ax.axis('off')
   
    fig.set_figwidth(figwidth)
    if fig_height is not None:
        fig.set_figheight(fig_height)
    
    #Create different column sizes
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, colWidths= col_widths, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    #mpl_table.scale(scale_row, 1)
    #Give layout to table
    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            
    #Move 
    cells = mpl_table.properties()["celld"] #Left allign text ofall body-cells
    for i in range(1, int(len(cells)/3)):
        cells[i, 1]._loc = 'left'
        cells[i, 0]._loc = 'left'
        cells[i, 2]._loc = 'left'
        
    [cells[0,i].set_height(cells[0,i].get_height() * 1.2) for i in range(0,3)]   # And make header a little larger
    
    return ax.get_figure()


def scale_figure_to_width(fig, new_width, font_size = 14):
    plt.ion()
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width*fig.dpi, bbox.height*fig.dpi
    aspect_ratio = height/width
    fig.set_figwidth(new_width)
    fig.set_figheight(new_width*aspect_ratio)
    #fig.set_fontsize(font_size)
    
    return fig

def set_global_fig_settings():
    #pyglet.font.add_file('C:/Users/Kees/pargo/pargo_fonts/Futura_Bold_ttf.ttf')
    #futura_bold = pyglet.font.load('Futura Bold')
    plt.rcParams['font.family'] = 'Futura Bk BT'
    plt.rc('font', size = 6)          # controls default text sizes
    plt.rc('axes', titlesize = 6)     # fontsize of the axes title
    plt.rc('axes', labelsize= 6)    # fontsize of the x and y labelsc
    plt.rc('xtick', labelsize= 6 )    # fontsize of the tick labels
    plt.rc('ytick', labelsize= 6 )    # fontsize of the tick labels
    plt.rc('legend', fontsize= 7 )    # legend fontsize
    plt.rc('figure', titlesize= 8 )  # fontsize of the figure title
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.facecolor'] = 'w'


# ratings
def find_rating(company, month_nr, year_nr):

    command_new_db = ("SELECT  "
    "SUM(CASE WHEN ratings.rating = 1 THEN 1 ELSE 0 END) AS rating_1, "
    "SUM(CASE WHEN ratings.rating = 2 THEN 1 ELSE 0 END) AS rating_2, "
    "SUM(CASE WHEN ratings.rating = 3 THEN 1 ELSE 0 END) AS rating_3, "
    "SUM(CASE WHEN ratings.rating = 4 THEN 1 ELSE 0 END) AS rating_4 "
    "FROM ratings " 
        "JOIN order_processes ON order_processes.uuid = ratings.order_process_id "
        "JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
    "WHERE "
        "EXTRACT (MONTH from order_processes.created_at) = " + str(month_nr) + " "
          " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " " #Don't forget the single quotations around company.
        "AND EXTRACT (YEAR from order_processes.created_at) = " + str(year_nr) + "; ")
    try:
        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        results_new_db_df = pd.DataFrame(results_new_db)
        results_new_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_new_db_df = pd.DataFrame(columns=['rating_1', 'rating_2', 'rating_3', 'rating_4'])

    command_old_db = ("SELECT "
                "SUM(CASE WHEN orders.ranking_pargo_point = 1 THEN 1 ELSE 0 END) AS rating_1, "
                "SUM(CASE WHEN orders.ranking_pargo_point = 2 THEN 1 ELSE 0 END) AS rating_2, "
                "SUM(CASE WHEN orders.ranking_pargo_point = 3 THEN 1 ELSE 0 END) AS rating_3, "
                "SUM(CASE WHEN orders.ranking_pargo_point = 4 THEN 1 ELSE 0 END) AS rating_4 "
            "FROM orders_status "
                "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
                "INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
            "WHERE "
                 " MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
                 " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
                "AND YEAR(orders_status.order_status_open) = " + str(year_nr) + " "
                "AND orders_status.order_status_collected IS NOT NULL; ")

    try:
        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        results_old_db_df = pd.DataFrame(results_old_db)
        results_old_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_old_db_df = pd.DataFrame(columns=['rating_1', 'rating_2', 'rating_3', 'rating_4'])

    results_old_db_df = results_old_db_df.fillna(0) #
    results_new_db_df = results_new_db_df.fillna(0)

    results_old_db_df.columns = results_old_db_df.columns.get_level_values(0)
    results_new_db_df.columns = results_new_db_df.columns.get_level_values(0)

    ratings_df =  results_new_db_df +  results_old_db_df
    return ratings_df
    
    
def find_locationtype_of_orders(company, month_nr, year_nr, location_type):
    
    command_new_db = ("SELECT  "
    "order_process_tracking_info.order_process_id AS order_id, "
    "pickup_points.location_type AS location_type "
    "FROM order_process_tracking_info " 
        "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
        "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
    # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
        "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
    "WHERE process_types.name = 'w2p' "
        "AND pickup_points.location_type = '" + str(location_type) + "' "
        "AND process_types_stages.name = 'confirmed' "
        "AND EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
        " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " " #Don't forget the single quotations around company.
        "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + "; ")

    try:
        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        results_new_db_df = pd.DataFrame(results_new_db)
        results_new_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_new_db_df = pd.DataFrame(columns=['order_id', 'location_type'])


    command_old_db = ("SELECT "
                "orders.ordersid AS order_id, "
                "pargopoint.location_type AS location_type "
            "FROM orders_status "
                "INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
                "INNER JOIN pargopoint ON pargopoint.pargopointid = orders.pargopointid "
                "INNER JOIN suppliers ON suppliers.suppliersid =orders.suppliersid "
            "WHERE "
                " MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
                " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
                "AND MONTH(orders_status.order_status_open) = " + str(month_nr) + " "
                "AND YEAR(orders_status.order_status_open) = " + str(year_nr) + " "
                "AND pargopoint.location_type  = '" + str(location_type) + "' "
                "AND orders_status.courier_pod IS NOT NULL; ")

    try:
        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        results_old_db_df = pd.DataFrame(results_old_db)
        results_old_db_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_old_db_df = pd.DataFrame(columns=['order_id', 'location_type'])

    results_old_db_df.columns = results_old_db_df.columns.get_level_values(0)
    results_new_db_df.columns = results_new_db_df.columns.get_level_values(0)

    results_df = pd.concat([results_old_db_df, results_new_db_df], sort = False)
    return results_df


def extract_courier_df_from_process_tracking_info(company, year_nr, month_nr):

    command = ("SELECT "
    "order_process_tracking_info.order_process_id AS order_id, "
    "order_process_tracking_info.created_at AS timestamp, "
    "process_types_stages.name AS courier_type_stage_name "
    "FROM order_process_tracking_info " 
        "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
        "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
        "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
        "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
    # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
    "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
    "WHERE process_types_stages.name = 'at_collection_point' "
    "AND order_process_tracking_info.order_process_id IN ( "
        "SELECT order_process_tracking_info.order_process_id "
        "FROM order_process_tracking_info " 
            "INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
            "INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
            "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
            "INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
        # "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
            "INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
        "WHERE process_types.name = 'w2p' "
            "AND process_types_stages.name = 'confirmed' "
            "AND EXTRACT (MONTH from order_process_tracking_info.created_at) = " + str(month_nr) + " "
            " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
            "AND EXTRACT (YEAR from order_process_tracking_info.created_at) = " + str(year_nr) + "); ")

    try:
        results, column_names = arbitrary_command_newdatabase(command)
        results_order_process_df = pd.DataFrame(results)
        results_order_process_df.columns = [column_names]
    except ValueError: # In case of empty dataframe, still output results and a dataframe that can be concatenated with other results
        results_order_process_df = pd.DataFrame(columns=['order_id', ' timestamp', 'courier_type_stage_name'])
    results_order_process_df = results_order_process_df.replace("at_collection_point", "delivered")
    results_order_process_df.columns = results_order_process_df.columns.get_level_values(0)
  
    return results_order_process_df

def make_year_month_array(nr_months_back = 5, year_nr = year_nr, month_nr = month_nr):
    year_month_array = np.zeros((nr_months_back, 2)) #Store month number and year here
    year_month_array_str = [] # Create a list with strings that can be used as xtic
    current_date = datetime.datetime(int(year_nr), int(month_nr), 1)
    for counter, decrement in enumerate(range(-nr_months_back + 1, 1)):
        decremented_month = increment_months(current_date, decrement)
        year_month_array[counter][0] = decremented_month.year
        year_str = str(decremented_month.year) #Isj a number so isj good
        year_month_array[counter][1] = decremented_month.month
        month_str = str(calendar.month_abbr[decremented_month.month])                           #convert to three character abbreviatins of month

        year_month_array_str.append([year_str, month_str])
    return year_month_array, year_month_array_str

def add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba):
    supl_filter_str = ""
    for counter, suppliers in enumerate(supplier_list_simba):
        if counter == 0:
            supl_filter_str += "AND (suppliers.reference = '" + str(suppliers) +"' "
        else:
            supl_filter_str += " OR suppliers.reference = '" + str(suppliers) + "' "
    supl_filter_str += " ) "
    return supl_filter_str 

def add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo):
    # filters the SQL to find the suppliers reference.

    supl_filter_str = ""
    for counter, suppliers in enumerate(supplier_list_pup_pargo):
        if counter == 0:
            supl_filter_str += "AND (suppliers.supplier_code = '" + str(suppliers) +"' "
        else:
            supl_filter_str += " OR suppliers.supplier_code = '" + str(suppliers) + "' "
    supl_filter_str += " ) "
    return supl_filter_str


def plot_hist_deliv(historical_deliv_data_df, year_month_array_str_deliv, title = ""):
    pargo_blue = np.asarray([24,120,185])/255
    x_axis_ym = [(str(year_month_array_str_deliv[i][1]) + " " + str(year_month_array_str_deliv[i][0])) for i in range(len(year_month_array_str_deliv))]
    hist_data_list = historical_deliv_data_df.values.tolist() #For some reason this has size 1 x n, not n x 1, so fix that in next line
    y_axis_ym_pre= [hist_data_list[i][-1] for i in range(len(hist_data_list))]

    y_axis_ym =  [round(i,2) for i in y_axis_ym_pre]

    y_pos = np.arange(len(year_month_array_str_deliv))
    fig = plt.figure(figsize=(5,2))
    plt.bar(y_pos, y_axis_ym, align='center', alpha=0.5, color = pargo_blue)
    plt.xticks(y_pos, x_axis_ym, rotation = 45, ha = 'right')
    plt.ylabel('Percentage of deliveries within SLA')
    plt.xlabel('Month')
    plt.ylim(0, max(y_axis_ym)*1.2)
    plt.title(title)

    for i in range(len(y_axis_ym)):
        plt.text(x = y_pos[i] , y = y_axis_ym[i]+ 0.05 * max(y_axis_ym), s = str(y_axis_ym[i]) + str("%"), ha = 'center', fontsize = 5)

    plt.show()
    return fig


###########################################################################################################################################


# MAINNN
print("If this is printed, Kernel has not crashed")
#Find timestamps from new database (Simba)
results_courier_df, results_order_process_df  = extract_timestamps_from_sql(company, month_nr, year_nr)
print("If this is printed, Kernel has not crashed")
#For some companies, information cannot be obtained by the courier
if (results_courier_df.empty == True and results_order_process_df.empty == False):
    results_courier_df = extract_courier_df_from_process_tracking_info(company, year_nr = year_nr, month_nr = month_nr)
    
#If there is information in the new database, merge and tidy the timestamps, so that delivery times can be calc'ed.
if ((results_courier_df.empty and results_order_process_df.empty) == False):
    df_timestamps_new_db = merge_and_tidy_timestamps(results_courier_df, results_order_process_df)
else: 
    df_timestamps_new_db  =  pd.DataFrame(columns=['order_id', 'status_confirmed', 'status_at_courier_for_deliv', 'status_completed'])

#And from old
df_timestamps_old_db = query_timestamps_mysql(company, year_nr = year_nr, month_nr = month_nr)

#For consistency, filter only the delivered parcels first
 
df_timestamps_old_db_delivered = df_timestamps_old_db[df_timestamps_old_db["status_completed"].notnull()]
df_timestamps_old_db_returned = df_timestamps_old_db[df_timestamps_old_db["status_completed"].isnull()]

df_timestamps_tot = pd.concat([df_timestamps_old_db_delivered, df_timestamps_new_db], sort = False)
df_timestamps_tot.columns = df_timestamps_tot.columns.get_level_values(0)

df_locations_main = find_locationtype_of_orders(company, month_nr, year_nr, location_type = "main")
df_locations_regional = find_locationtype_of_orders(company, month_nr, year_nr, location_type = "regional")

df_timestamps_regional = df_timestamps_tot.merge(df_locations_regional, on = "order_id")
df_timestamps_main = df_timestamps_tot.merge(df_locations_main, on = "order_id")
#Now merge old and new database values


#Extract deliverytimes
deliv_times_np_days = extract_deliv_times_in_days(df_timestamps_tot)
deliv_times_np_days_main = extract_deliv_times_in_days(df_timestamps_main)
deliv_times_np_days_regional = extract_deliv_times_in_days(df_timestamps_regional)

#Extract collection dates
collected_date, collected_days, weekdays, collection_weekdays = extract_collected_dates_and_weekdays(df_timestamps_tot)

#Extract some parameters and use those to make the horizontal bar with delivery times versus number of parcels
nr_of_delivered_parcels, day_labels, nr_of_suppliers,  unique_days, unique_days_counts, x_bar, y_bar = extract_parameters(deliv_times_np_days, df_timestamps_tot, stack_from_x_days = 6)

nr_of_delivered_parcels_main, day_labels, nr_of_suppliers, unique_days_main, unique_days_counts_main, x_bar_main, y_bar_main = extract_parameters(deliv_times_np_days_main, df_timestamps_main, stack_from_x_days = 6)

nr_of_delivered_parcels_regional, day_labels, nr_of_suppliers, unique_days_regional, unique_days_counts_regional, x_bar_regional, y_bar_regional = extract_parameters(deliv_times_np_days_regional, df_timestamps_regional, stack_from_x_days = 6)
print("We're at flag 2 now")


#Calc SLAs
main_within_sla = (deliv_times_np_days_main <= 4).sum()
regional_within_sla = (deliv_times_np_days_regional <= 6).sum()
tot_within_sla = main_within_sla + regional_within_sla
#Make graph with info on what weekdays peoples collect their parcels

#Make graph for checking the number of days that customers take to collect a parcel
collection_np_after_x_days = extract_collection_times_in_days(df_timestamps_tot)
nr_of_collected_parcels, day_labels, nr_of_suppliers, unique_col_days, unique_col_counts, x_bar_col, y_bar_col = extract_parameters(collection_np_after_x_days, df_timestamps_tot, stack_from_x_days = 6)

#Plot on which weekdays a parcel is collected
#Extract params for collection weekdays and hours, plot them and make pie chart for office hours
collection_hours, collection_dates, collection_times = extract_col_times_hours_dates(df_timestamps_tot['status_completed'])

nr_of_returned_parcels_new_db = find_nr_of_returned_parcels_sql(company, month_nr, year_nr)
nr_of_returned_parcels = nr_of_returned_parcels_new_db + df_timestamps_old_db_returned.shape[0] #From old database + new database


historical_data, year_month_array_str = extract_historial_parcel_deliveries_data(company, nr_months_back = nr_months_back, month_nr = month_nr, year_nr = year_nr)


tab = deliveries_per_pup_sql(company, month_nr, year_nr)

#find ratings
ratings_df = find_rating(company, month_nr, year_nr)

#Find current states of parcels
nr_parcels_collected, nr_parcels_returned, nr_parcels_in_stock = find_current_states_parcels(company , year_nr , month_nr )
print("Flag 3: Almost finished")
#Find collected vs not_collected numbers


collected_not_collected_df, year_month_array_str_col_rate = find_collected_not_collected_prev_x_months(company, year_nr = year_nr, month_nr = month_nr)

print("we are at collected_not_collected_df")
historical_deliv_data_df, year_month_array_str_deliv = extract_historical_delivery_rates(company, year_nr = year_nr, month_nr = month_nr, nr_months_back = 6)

#Last month is not calculated well because only order_id's from the prev month are taken into account.
historical_deliv_data_df['tot_within_sla'].iloc[historical_deliv_data_df.index[-1]] = tot_within_sla
historical_deliv_data_df['tot_outside_sla'].iloc[historical_deliv_data_df.index[-1]] = nr_of_delivered_parcels

print("we are at historical['tot_outside_sla']")
historical_deliv_data_df['percentage_sla'].iloc[historical_deliv_data_df.index[-1]] = tot_within_sla/nr_of_delivered_parcels * 100 


print("we are done with this cell")


############################################################################################################################

print("Kernel didn't die")


### This Notebook plots all the Charts and Graphs. 

set_global_fig_settings()

hist_bar_chart = plot_historical_parcel_deliveries_data(historical_data, year_month_array_str, title = 'AN OVERVIEW OF THE TOTAL NUMBER OF DELIVERED PARCELS PER MONTH')
hist_bar_chart_scaled = scale_figure_to_width(hist_bar_chart, new_width = 6.5)
hist_bar_chart_svg = from_fig_to_svg(hist_bar_chart_scaled)
#scaled_hist_bar_chart_svg = scale_image(hist_bar_chart_svg, scaling_factor=0.35)

pie_parcel_main_regional = pie_chart(title = "SPLIT MAIN/REGIONAL", data = [nr_of_delivered_parcels_regional, nr_of_delivered_parcels_main], labels = ["Regional", "Main"])
pie_parcel_main_regional_scaled = scale_figure_to_width(pie_parcel_main_regional, new_width = 2)
pie_parcel_main_regional_svg =  from_fig_to_svg(pie_parcel_main_regional_scaled )


#Pie chart status
pie_parcel_status = pie_chart_parcel_status(nr_parcels_collected, nr_parcels_returned, nr_parcels_in_stock, "CURRENT PARCEL STATUS") 
pie_parcel_status_scaled = scale_figure_to_width(pie_parcel_status , new_width = 2)
pie_parcel_status_svg =  from_fig_to_svg(pie_parcel_status_scaled )
#scaled_pie_parcel_status_svg = scale_image(pie_parcel_status_svg, scaling_factor = 0.3)


#Plot collection rates

fig_collected_not_collected = plot_collected_not_collected_df(collected_not_collected_df, 'COLLECTION RATES PER MONTH')
fig_collected_not_collected_scaled = scale_figure_to_width(fig_collected_not_collected, new_width = 5)
fig_collected_not_collected_svg =  from_fig_to_svg(fig_collected_not_collected_scaled)


#Nr of deliveries per PUP TOP 10
rendered_table = render_mpl_table(tab)
rendered_table_scaled = scale_figure_to_width(rendered_table , new_width = 4)
top_pups_svg = from_fig_to_svg(rendered_table_scaled)

#Horizontal bar delivery times average
plot_deliveries = make_h_bar(labels = day_labels, segments = nr_of_suppliers, ylabel = "Business days", dat = y_bar, title = "DELIVERY TIMES") 
plot_deliveries_scaled = scale_figure_to_width(plot_deliveries, new_width = 3)
delivery_times_svg =  from_fig_to_svg(plot_deliveries_scaled)

#Piechart regional within SLA
pie_parcel_regional_sla = pie_chart(title = "REGIONAL DELIVERY TIMES IN 5 DAYS", data = [regional_within_sla, nr_of_delivered_parcels_regional - regional_within_sla], labels = ["Within 5 days", "Longer than 5 days"])
pie_parcel_regional_sla_scaled = scale_figure_to_width(pie_parcel_regional_sla, new_width = 2)
pie_parcel_regional_sla_svg =  from_fig_to_svg(pie_parcel_regional_sla_scaled )

#Piechart main within SLA
pie_parcel_main_sla = pie_chart(title = "MAIN DELIVERY TIMES IN 3 DAYS", data = [main_within_sla, nr_of_delivered_parcels_main - main_within_sla], labels = ["Within 3 days", "Longer than 3 days"])
pie_parcel_main_sla_scaled = scale_figure_to_width(pie_parcel_main_sla, new_width = 2)
pie_parcel_main_sla_svg =  from_fig_to_svg(pie_parcel_main_sla_scaled )


#Piechart total within SLA
pie_parcel_tot_sla = pie_chart(title = "DELIVERY TIMES", data = [tot_within_sla, nr_of_delivered_parcels_main + nr_of_delivered_parcels_regional - tot_within_sla], labels = ["Within SLA", "Outside of SLA"])
pie_parcel_tot_sla_scaled = scale_figure_to_width(pie_parcel_tot_sla, new_width = 2)
pie_parcel_tot_sla_svg =  from_fig_to_svg(pie_parcel_tot_sla_scaled)

#Horizontal bar delivery times main
plot_deliveries_main = make_h_bar(labels = day_labels, segments = nr_of_suppliers, dat = y_bar_main, ylabel = "Business days", title = "DELIVERY TIMES IN MAIN") 
plot_deliveries_main_scaled = scale_figure_to_width(plot_deliveries_main, new_width = 3)
delivery_times_main_svg =  from_fig_to_svg(plot_deliveries_main_scaled)

#Horizontal bar delivery time regional
plot_deliveries_reg = make_h_bar(labels = day_labels, segments = nr_of_suppliers,  dat = y_bar_regional, ylabel = "Business days", title = "DELIVERY TIMES IN REGIONAL", color_id = 1) 
plot_deliveries_reg_scaled = scale_figure_to_width(plot_deliveries_reg, new_width = 3)
delivery_times_reg_svg =  from_fig_to_svg(plot_deliveries_reg_scaled)

#Pre-after working hours delivery
piechart_office_hours = piechart_collection_after_pre_work(collection_hours, "COLLECTION OFFICE HOURS")
piechart_office_hours_scaled = scale_figure_to_width(piechart_office_hours, new_width =2)
piechart_office_hours_svg =  from_fig_to_svg(piechart_office_hours_scaled)
#scaled_piechart_office_hours_svg = scale_image(piechart_office_hours_svg, scaling_factor = 0.3)

# Nr of days that customers take to collect
plot_collections = make_h_bar(fig_size = (5,2), labels = day_labels, segments = nr_of_suppliers, dat =y_bar_col, 
                              title = "INFORMATION REGARDING THE NUMBER OF DAYS AFTER WHICH CUSTOMERS COLLECT PARCELS")
plot_collections_scaled = scale_figure_to_width(plot_collections, new_width = 5)
collection_hbar_svg =  from_fig_to_svg(plot_collections)

#Plot collection times of days
plot_collection_times_of_day = plot_collection_time_of_days(collection_hours, title = "INFORMATION ON THE TIMES OF THE DAY AT WHICH CUSTOMERS PICK UP PARCELS")
plot_collection_times_of_day_scaled = scale_figure_to_width(plot_collection_times_of_day, new_width = 5)
plot_collection_times_of_day_svg =  from_fig_to_svg(plot_collection_times_of_day_scaled)
#scaled_collection_times_of_day_svg = scale_image(plot_collection_times_of_day_svg, scaling_factor = 0.35)


#Create fig collection weekdays
fig_collection_weekdays = plot_collection_weekdays(collection_weekdays, title = 'COLLECTION PER WEEKDAY')
fig_collection_weekdays_scaled = scale_figure_to_width(fig_collection_weekdays, new_width = 5)
fig_collection_weekdays_svg =  from_fig_to_svg(fig_collection_weekdays_scaled)
#scaled_fig_collection_weekdays_svg = scale_image(fig_collection_weekdays_svg, scaling_factor = 0.35)

#Create fig collection per day of the month

fig_collection_dayofmonth = plot_collection_monthdays(collected_days, "THE NUMBER OF COLLECTIONS PER DAY OF THE MONTH")
fig_collection_dayofmonth_scaled = scale_figure_to_width(fig_collection_dayofmonth, new_width = 7.5)
fig_collection_dayofmonth_svg =  from_fig_to_svg(fig_collection_dayofmonth_scaled)

#Create large table with delivs of pickup points
tab_large = deliveries_per_pup_sql(company, month_nr, year_nr, show_top_x = 60)
rendered_tab_large = render_mpl_table(tab_large,  fig_height = 12)
rendered_tab_large_scaled = scale_figure_to_width(rendered_tab_large , new_width = 7)
rendered_tab_large_svg = from_fig_to_svg(rendered_tab_large_scaled)


#if len(ratings_df) >= 30: #Only make image if <30 ratings 
csat_pie = pie_circle_chart(title = "CSAT (CUSTOMERS SATISFACTION) SCORE", data = [int(ratings_df['rating_1'].iloc[0]), ratings_df['rating_2'].iloc[0],ratings_df['rating_3'].iloc[0], ratings_df['rating_4'].iloc[0]], labels = ["1", "2", "3", "4"])
csat_pie_scaled = scale_figure_to_width(csat_pie , new_width = 2)
csat_pie_svg = from_fig_to_svg(csat_pie_scaled)


#Historical delivery percentage within SLA rates
fig_hist_delivs = plot_hist_deliv(historical_deliv_data_df, year_month_array_str_deliv, title = 'DATA ON THE HISTORICAL NUMBER OF DELIVERIES WITHIN SLA')
fig_hist_delivs_scaled = scale_figure_to_width(fig_hist_delivs, new_width = 5)
fig_hist_delivs_svg = from_fig_to_svg(fig_hist_delivs_scaled)


##################################################################################################################

month_nr_to_str_dict = {1: "January", 2: "February", 3: "March", 4: "April" , 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"} 

pdfmetrics.registerFont(TTFont('Futura', '/home/abulele/Pictures/pargo_fonts/16020_FUTURAM.ttf'))
pdfmetrics.registerFont(TTFont('Open Sans', '/home/abulele/Pictures/pargo_fonts/open_sans/OpenSans-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Futura Bold', '/home/abulele/Pictures/pargo_fonts/Futura_Bold_ttf.ttf'))
fancy_header_line_svg = svg2rlg("/home/abulele/Pictures/pargo_figs/DataReport_BarComponent-11.svg") #Import fancy header bar thingy



pdf_path = "/home/abulele/Documents/pargo_datasheets/new_layout_" + str(company) + "_" + str(month_nr) + "_" + str(year_nr)  + "_datasheet.pdf"
c = canvas.Canvas(pdf_path) #Create a pdf object
page_width  = defaultPageSize[0] #595
page_height = defaultPageSize[1] #841
pargo_blue = np.asarray([24,120,185])/255

x_min = 15
y_min = 10
textbox_width = 97 #Empirically found that 80 characters max should be used to fit the line
#Write header. 


def write_front_page():
    #Add background
    front_cover_svg = svg2rlg("/home/abulele/Pictures/pargo_figs/DataReportCover-02.svg")
    front_cover_scaled_svg = scale_image_to_width(front_cover_svg,  to_width = page_width)
    renderPDF.draw(front_cover_scaled_svg , c, 0, 0)
    
    #Add texts and line, starting at bottom corner
    left_bound_text = 40
    c.setFillColorRGB(1,1,1) #Make everything white
    c.setFont('Futura', 15, leading = None)
    c.drawString(left_bound_text, 60, str(month_nr_to_str_dict[int(month_nr)]).upper() + " " + str(int(year_nr)))
    
    c.setFont('Futura', 18, leading = None)
    c.drawString(left_bound_text, 60 + 22, "FOR " + str(company).upper())
    
    c.setFont('Futura Bold', 28, leading = None)
    c.drawString(left_bound_text, 60 + 22 + 18, "MONTHLY REPORT")
    height_line = 60 + 22 + 18 + 33
    
    c.setStrokeColorRGB(1,1,1)
    c.setFillColorRGB(1,1,1)

    c.roundRect(x = 32, y = height_line , width = 370 - 32, height = 1, radius = 1, stroke = 1, fill =  1)
    
    c.setFont('Futura Bold', 60,  leading = None)
    c.drawString(left_bound_text, 60 + 22 + 18 + 33 +  16, "PARGO")
    
    c.setFillColorRGB(0,0,0) # Reverse text colour to default.

    c.showPage()
    
def write_header( page_name, lines, company = company, month_nr = month_nr, year_nr = year_nr):
    renderPDF.draw(fancy_header_line_svg , c, page_width * 0.5 - 0.5 * fancy_header_line_svg.width, page_height - 50)
    c.setFillColorRGB(24/255,120/255,185/255)
    c.setFont('Futura Bold', 30, leading = None)
    c.drawString(x_min, page_height - 105 , page_name) 

    textobject = c.beginText(15, page_height - 135)
    textobject.setFillColorRGB(0, 0, 0)
    textobject.setFont('Open Sans', 12, leading = None)
    #Write text
    
    
    for line in lines:
        textobject.textLine(line)
    c.drawText(textobject)
    
def write_page_overview(page_name = "OVERVIEW"):
    lines = ("The statistics below displays an overview of the services provided by Pargo for " + str(company) + " for the month "  
    "of " + str(month_nr_to_str_dict[int(month_nr)]) + " " + str(int(year_nr)) + ". The data sets include information on deliveries, collections and Pick-up Points." )
    lines = wrap(lines, textbox_width)
    print(lines)
    write_header(lines = lines, page_name = page_name)

    #Write page 1

    #Draw 1.1 horizontal bar chart
    renderPDF.draw(hist_bar_chart_svg , c, page_width* 0.5 - hist_bar_chart_svg.width/2, page_height - 205 - hist_bar_chart_svg.height)

    #Most used pups table
    renderPDF.draw(top_pups_svg, c, page_width* 0.5 - top_pups_svg.width/2, page_height - 205 - hist_bar_chart_svg.height - 55 - top_pups_svg.height)


    #Triple entente of pie charts
    renderPDF.draw(pie_parcel_tot_sla_svg, c, x_min, y_min + 5)
    renderPDF.draw(pie_parcel_status_svg, c, 0.5*page_width - pie_parcel_status_svg.width/2, y_min + 5)
    renderPDF.draw(pie_parcel_main_regional_svg, c, page_width - x_min - pie_parcel_main_regional_svg.width , y_min + 5) #This actually creates the page. Useful for multi-pages
    c.showPage()
    


def write_pages_delivery_times(): 
    lines = ("The statistics below displays an overview of the services provided by Pargo for " + str(company) + " for the month " 
    "of " + str(month_nr_to_str_dict[int(month_nr)]) + " " + str(int(year_nr)) + ". The data sets include information on deliveries: "
    "delivery times to both regional and main areas, as well as the delivery of parcels within Pargo’s SLA.")
    # column1_height = 0.75  * page_height   #Line height is at 0.9. Subtract 0.05 on both ends for margins. Split the remainder 0.85 in threes
    lines = wrap(lines, textbox_width)
    column2_height = 0.65  * page_height
    column3_height = 0.35  * page_height

    column1_width = 0.28 * page_width
    column2_width = 0.73 * page_width

    
    #Page 1
    write_header(page_name = "DELIVERY TIMES", lines = lines)

    #Average delivery times hbar plot
    renderPDF.draw(delivery_times_svg, c, column2_width - 0.5 * delivery_times_svg.width, page_height - 205 - delivery_times_svg.height)

    #Within/outside SLA on average Pie
    renderPDF.draw(pie_parcel_main_regional_svg, c, column1_width - 0.5*pie_parcel_main_regional_svg.width, page_height - 205 - pie_parcel_main_regional_svg.height)

    renderPDF.draw(fig_hist_delivs_svg, c, 0.5 * page_width - fig_hist_delivs_svg.width/2, y_min + 150) 
    
    c.showPage()
    #Page 2
    write_header(page_name = "DELIVERIES", lines = lines)
    
    #Regional within/outside sla pie-chart 
    renderPDF.draw(pie_parcel_regional_sla_svg, c, column1_width - pie_parcel_regional_sla_svg.width/2, column2_height -pie_parcel_regional_sla_svg.height/2) 

    #Regional delivery times h-bar
    renderPDF.draw(delivery_times_reg_svg, c, column2_width - delivery_times_reg_svg.width/2, column2_height - delivery_times_reg_svg.height/2) 

    #Delivery times main h-bar
    renderPDF.draw(delivery_times_main_svg, c, column2_width - delivery_times_main_svg.width/2 , column3_height - delivery_times_main_svg.height/2)

    #Delivery times main SLA pie-chart
    renderPDF.draw(pie_parcel_main_sla_svg, c, column1_width  - pie_parcel_main_sla_svg.width/2, column3_height - pie_parcel_main_sla_svg.height/2) 

    c.showPage()
    
def write_pages_collections(): 
    lines = ("The statistics below displays an overview of the services provided by Pargo for " + str(company) + " for the month " 
    "of " + str(month_nr_to_str_dict[int(month_nr)]) + " " + str(int(year_nr)) + ". The data sets include information on collections: collections per day of month, " 
    "collection times within a day, and the number of days it takes customers to collect.")
    lines = wrap(lines, textbox_width)
    #Page 1
    write_header(page_name = "COLLECTIONS", lines = lines)
    renderPDF.draw(fig_collection_weekdays_svg, c, 0.5* (page_width - fig_collection_weekdays_svg.width) , page_height - 205 - fig_collection_weekdays_svg.height)
    

    #Collection rates
    renderPDF.draw(fig_collected_not_collected_svg , c,  0.5 * page_width - fig_collected_not_collected_svg.width/2, page_height - 205 - fig_collection_weekdays_svg.height - 5 - fig_collected_not_collected_svg.height)
    #Tripe pie entente
    renderPDF.draw(piechart_office_hours_svg, c, x_min, y_min + 5)
    
    if sum(ratings_df.iloc[0]) >= 30: #If more than 30 peoples have given a rating, include the CSAT image
        renderPDF.draw(csat_pie_svg, c, 0.5*page_width - csat_pie_svg.width/2, y_min + 5 + piechart_office_hours_svg.height - csat_pie_svg.height )
    
    renderPDF.draw(pie_parcel_status_svg, c, page_width - x_min - pie_parcel_status_svg.width , y_min + 5)
        
    c.showPage()
    
    #Page 2
    write_header(page_name = "COLLECTIONS", lines = lines)
    renderPDF.draw(fig_collection_dayofmonth_svg , c, 0.5 * page_width - fig_collection_dayofmonth_svg.width/2 , page_height - 205 - fig_collection_dayofmonth_svg.height )
    renderPDF.draw(plot_collection_times_of_day_svg, c, 0.5 * page_width - plot_collection_times_of_day_svg.width/2, page_height - 205 - fig_collection_dayofmonth_svg.height - 20 - plot_collection_times_of_day_svg.height)
    renderPDF.draw(collection_hbar_svg,c, page_width * 0.5 -collection_hbar_svg.width/2, y_min)
    
    c.showPage()
    

def write_page_pickup_points():
   
    lines = ("The statistics below displays an overview of the services provided by Pargo for " + str(company) + " for the month "
    "of " + str(month_nr_to_str_dict[int(month_nr)]) + " " + str(int(year_nr)) + ". The data set includes information on Pick-up Points: which Pick-up Point handled " 
    "the largest volume of " + str(company) + " parcels.")
    lines = wrap(lines, textbox_width)
    #Page 1
    write_header(page_name = "PICKUP-POINTS", lines = lines)
    
    #renderPDF.draw(scaled_rendered_tab_large_svg_2, c, -page_width*0.3, page_height * 0.0)
    renderPDF.draw(rendered_tab_large_svg, c, page_width*0.5 -  rendered_tab_large_svg.width/2,  page_height - 205 - rendered_tab_large_svg.height)
    c.showPage()
    
def write_page_appendix():
  
    lines = ["Figures of the previous pages are elaborated in more details here."]
    write_header(page_name = "APPENDIX", lines = lines)
    #Figs of page 1
    lines_app = []
    lines_app.append("FIGURES OF PAGE 'OVERVIEW'. ")
    lines_app.append("AN OVERVIEW OF THE TOTAL NUMBER OF DELIVERED PARCELS PER MONTH: this bar chart shows the total number of parcels")
    lines_app.append("that have been delivered to any store for " + str(company) + "via the services of Pargo over the past months.")
    lines_app.append(" ")
    
    lines_app.append("TOP 10 STORES TABLE: this table shows the stores that the largest volumes of parcels were delivered to.")
    lines_app.append(" ")
    
    lines_app.append("DELIVERY TIMES: this pie chart shows the percentages of parcels that were delivered within and outside of")
    lines_app.append("Pargo's Service-Level Agreement.")
    lines_app.append(" ")
    
    lines_app.append("CURRENT PARCEL STATUS: this pie chart shows the current status of parcels. Thet status 'collected' refers to parcels that ") 
    lines_app.append("have been collected by customers, 'in stock' refers to parcels that still lay at pick-up points and 'Returned' refers")
    lines_app.append("to parcels that have been returned to the warehouse.")
    lines_app.append(" ")
    
    lines_app.append("SPLIT MAIN/REGIONAL: this pie chart shows the percentages of parcels that were sent to pick-up points that are")
    lines_app.append("located in either main or in regional areas.")
    lines_app.append(" ")
    #Figs of page 2.1
    lines_app.append("SPLIT MAIN/REGIONAL: this pie chart shows the percentages of parcels that were sent to pick-up points that are")
    lines_app.append("located in either main or in regional areas.")
    lines_app.append(" ")
    
    lines_app.append("DELIVERY TIMES: this bar graph shows the number of business days after which Pargo's courier services ")
    lines_app.append("delivered their parcels.")
    lines_app.append(" ")
    
    lines_app.append("DATA ON THE HISTORICAL NUMBER OF DELIVERIES WITHIN SLA: this bar chart shows the percentages of parcels")
    lines_app.append("that were delivered within Pargo's Service Level Agreement over previous months.")
    lines_app.append(" ")
    #figs of page 2.1
    lines_app.append("REGIONAL DELIVERY TIMES IN 5 DAYS: This pie chart shows the percentage of parcels that were delivered to") 
    lines_app.append("stores in regional areas within 5 days, as conform Pargo's Service-Level Agreement.")
    lines_app.append(" ")
    
    lines_app.append("DELIVERY TIMES IN REGIONAL: this bar graph shows the number of business days after which Pargo's courier services ")
    lines_app.append("delivered their parcels to stores in regional areas.")
    lines_app.append(" ")
    
    lines_app.append("MAIN DELIVERY TIMES IN 5 DAYS: This pie chart shows the percentage of parcels that were delivered to")
    lines_app.append("stores in regional areas within 3 days, as conform Pargo's Service-Level Agreement.")     
    lines_app.append(" ")
    
    lines_app.append("DELIVERY TIMES IN MAIN: this bar graph shows the number of business days after which Pargo's courier services ")
    lines_app.append("delivered their parcels to stores in main areas.")
    lines_app.append(" ")
    
    #figs of page 3.1
    lines_app.append("COLLECTION PER WEEKDAY: This bar chart shows the number of parcels that were picked up on any of the weekdays")
    lines_app.append(" ")
    
    lines_app.append("COLLECTION RATES PER MONTH: This bar chart shows the percentage of parcels that have been collected by customers")
    lines_app.append("versus the number of parcels that have been returned to the warehouse.")
    lines_app.append(" ")
    
    lines_app.append("COLLECTION OFFICE HOURS: this pie chart shows the percentage of parcels that have been collected by customers ")
    lines_app.append("within and outside of business hours (9am to 5 pm).")
    lines_app.append(" ")
    
    lines_app.append("CURRENT PARCEL STATUS: this pie chart shows the current status of parcels. Thet status 'collected' refers to parcels that ")
    lines_app.append("have been collected by customers, 'in stock' refers to parcels that still lay at pick-up points and 'Returned' refers")
    lines_app.append("to parcels that have been returned to the warehouse.")
    lines_app.append(" ")
    
    #Figs of page 3.2
    lines_app.append("THE NUMBER OF COLLECTIONS PER DAY OF THE MONTH: this bar chart shows the number of parcels that were collected by ")
    lines_app.append("customers on each day of the month.")
    lines_app.append(" ")
    
    lines_app.append("INFORMATION ON THE TIMES OF THE DAY AT WHICH CUSTOMERS PICK UP PARCELS: this graph shows the hours of the day at ")
    lines_app.append("which customers collect parcels")
    lines_app.append(" ")
    
    lines_app.append("INFORMATION REGARDING THE NUMBER OF DAYS AFTER WHICH CUSTOMERS COLLECT PARCELS: this bar chart shows the number of ")
    lines_app.append("days after which customers collect parcels that have been delivered to stores.") 
    lines_app.append(" ")
    
    #Figs of page 4
    lines_app.append("TOP 60 STORES TABLE: this table shows the stores that the largest volumes of parcels were delivered to.")
    lines_app.append(" ")
    
    textobject = c.beginText(15, page_height - 165)
    textobject.setFillColorRGB(0, 0, 0)
    textobject.setFont('Open Sans', 9, leading = None)
    #Write text
    
    for line in lines_app:
        textobject.textLine(line)
    c.drawText(textobject)
    c.showPage()

write_front_page()
write_page_overview()
write_pages_delivery_times()
write_pages_collections()
write_page_pickup_points()
#write_page_appendix()                                                                
c.save()


###################################################################################################

plot_hist_deliv(historical_deliv_data_df, year_month_array_str_deliv, title = 'DATA ON THE HISTORICAL NUMBER OF DELIVERIES WITHIN SLA')
fig_collection_dayofmonth = plot_collection_monthdays(collected_days, "THE NUMBER OF COLLECTIONS PER DAY OF THE MONTH")


unique_day_of_col, unique_day_of_col_counts = np.unique(collection_dates,  return_counts=True)
all_days = np.linspace(1, calendar.monthrange(int(year_nr), int(month_nr))[1] , calendar.monthrange(int(year_nr), int(month_nr))[1]) #monthrange returns nr of days in a month. Add 1 to correct for indexing
#Make some ticks for each individual day later

#Make sure that days with zero sales are still included
for i in range(1,  calendar.monthrange(int(year_nr), int(month_nr))[1]  + 1):
    if i not in unique_day_of_col:
        unique_day_of_col_counts = np.insert(unique_day_of_col_counts, i - 1 , 0)
plt.figure(figsize = (5,2))
plt.plot(all_days, unique_day_of_col_counts)
plt.title("Collections per day of the month")
plt.ylabel("Parcels")
plt.show()

year_month_array = np.zeros((nr_months_back, 2)) #Store month number and year here
year_month_array_str = [] # Create a list with strings that can be used as xtic
current_date = datetime.datetime(int(year_nr), int(month_nr), 1)
for counter, decrement in enumerate(range(-nr_months_back, 0)):
    decremented_month = increment_months(current_date, decrement)
    year_month_array[counter][0] = decremented_month.year
    year_str = str(decremented_month.year) #Isj a number so isj good
    year_month_array[counter][1] = decremented_month.month
    month_str = str(calendar.month_abbr[decremented_month.month])                           #convert to three character abbreviatins of month

    year_month_array_str.append([year_str, month_str])

#QUERY FROM OLD DATABASE
# Okaye so you column names have to contain at least one letter. Use 'ym_year_month' format now, where year, month are numbers.

#We have to select a lot. In str_middle select which columns we actually want (sums of number of parcels created in a certain month, year)
#. End this part with the last month, and note to exclude the comma or SQL crashes.
#In str_end grab the data that we want to select from.
str_start = "Select "

str_middle = ""
for i in range(0, year_month_array.shape[0] - 1):
    str_middle = (str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END) "
    #"AS " + " jan_2019 "+ ", ")
    "AS ym_" + str(int(year_month_array[i][0])) + "_" + str(int(year_month_array[i][1])) + " , ")

    #str_middle = str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' THEN 1 ELSE 0 END) AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END), "
str_middle = (str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[-1][0])) + "' AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[-1][1])) + "' THEN 1 ELSE 0 END) "
"AS ym_" + str(int(year_month_array[-1][0])) + "_" + str(int(year_month_array[-1][1])) + "  ")

str_end = ("FROM order_process_tracking_info " 
"INNER JOIN process_types ON process_types.uuid = order_process_tracking_info.process_type_id " 
"INNER JOIN order_processes ON order_processes.uuid = order_process_tracking_info.order_process_id " 
"INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
"INNER JOIN process_types_stages ON process_types_stages.uuid = order_process_tracking_info.new_process_types_stages_id "  
# "INNER JOIN courier_tracking_info ON courier_tracking_info.order_process_id = order_processes.uuid "
"INNER JOIN pickup_points ON pickup_points.uuid = order_processes.to_id "
"WHERE process_types.name = 'w2p' "
    "AND process_types_stages.name = 'confirmed' "                                       
    " " + str(add_filter_on_suppliers_strings_simba(supplier_list_simba = supplier_list_simba)) + " "
    "AND order_processes.current_process_stage <> 'f8c666c1-04a3-4e18-ac4e-eee00c572780' " #<> is the IS NOT EQUAL TO operator. The uuid refers to w2p cancelled
";" )

command_new_db = str_start + str_middle + str_end

results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)

#results, column_names = arbitrary_command(command)

#'''
results_new_db_df = pd.DataFrame(results_new_db)
results_new_db_df.columns = [column_names]

results_new_db_df.columns = results_new_db_df.columns.get_level_values(0)
results_new_db_df = results_new_db_df.fillna(0)

#SELECT ALL THE "OPENED" OF A YEAR/MONTH, WHERE TIMESTAMP FINAL IS NOT NULL
str_start = "Select "

str_middle = ""
for i in range(0, year_month_array.shape[0] - 1):
    str_middle = (str_middle + " SUM(CASE WHEN YEAR(orders_status.order_status_open) = '" + str(int(year_month_array[i][0])) + "' AND MONTH(orders_status.order_status_open) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END) "
    #"AS " + " jan_2019 "+ ", ")
    "AS ym_" + str(int(year_month_array[i][0])) + "_" + str(int(year_month_array[i][1])) + " , ")

    #str_middle = str_middle + " SUM(CASE WHEN EXTRACT (YEAR from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][0])) + "' THEN 1 ELSE 0 END) AND EXTRACT (MONTH from order_process_tracking_info.created_at) = '" + str(int(year_month_array[i][1])) + "' THEN 1 ELSE 0 END), "
str_middle = (str_middle + " SUM(CASE WHEN YEAR(orders_status.order_status_open ) = '" + str(int(year_month_array[-1][0])) + "' AND MONTH(orders_status.order_status_open) = '" + str(int(year_month_array[-1][1])) + "' THEN 1 ELSE 0 END) "
"AS ym_" + str(int(year_month_array[-1][0])) + "_" + str(int(year_month_array[-1][1])) + "  ")

str_end = ("FROM orders_status "
"INNER JOIN orders ON orders.ordersid = orders_status.ordersid "
"INNER JOIN suppliers ON suppliers.suppliersid = orders.suppliersid "
"WHERE orders_status.order_status_open IS NOT NULL "
 " " + str(add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo = supplier_list_pup_pargo)) + " "
";")
command_old_db = str_start + str_middle + str_end

results_old_db, column_names = send_command_to_mysql_database(command_old_db)
results_old_db_df = pd.DataFrame(results_old_db)
results_old_db_df.columns = [column_names]
results_old_db_df.columns = results_old_db_df.columns.get_level_values(0)
results_old_db_df =results_old_db_df.fillna(0)

results_df = results_old_db_df + results_new_db_df