"""
Author: Abulele Mditshwa
Student No: Junior Data Scientist
email: abulele@capeai.com
The objective of this code is to have functions that plot collections for PUP (Pickup UP Point).
"""

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


# Don't take business days into account when calculating collection times
def extract_collection_times_in_days(df_timestamps_tot):
    # isolate dates of stuff where dates are relevant
    collected_date = [df_timestamps_tot['status_completed'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]
    at_col_date = [df_timestamps_tot['status_at_collection_point'].iloc[i].date() for i in
                   range(df_timestamps_tot.shape[0])]

    collected_date = np.asarray(collected_date)
    at_col_date = np.asarray(at_col_date)

    collection_times_days = [calc_daydiff(at_col_date[i], collected_date[i], corrected=False) for i in
                             range(collected_date.shape[0])]

    collection_times_np_days = np.asarray(collection_times_days)
    collection_np_after_x_days = collection_times_np_days[
        collection_times_np_days >= 0]  # Just to make sure that only correct values are taken
    return collection_np_after_x_days


# Exxtract information that's relevant when calculating the days of the month at which people pick up parcles
# And the day of the week (monday, tuesday etc)
def extract_collected_dates_and_weekdays(df_timestamps_tot):
    """
    1. Takes a DataFrame with collection timestamps
    2. Uses the python date function to get extra info such as date , week, month etc.
    3. Returns all those values that are extracted.
    """
    collected_date = [df_timestamps_tot['status_completed'].iloc[i].date() for i in range(df_timestamps_tot.shape[0])]
    collected_date = np.asarray(collected_date)
    collection_weekdays = [collected_date[i].weekday() for i in range(
        collected_date.shape[0])]  # INDEXING STARTS AT 0 HERE!! (as opposed to 1 in .isoweekday())
    collected_days = [collected_date[i].day for i in range(collected_date.shape[0])]
    weekday_str_to_index = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday",
                            6: "Sunday"}
    weekdays = [weekday_str_to_index[i] for i in range(len(weekday_str_to_index))]
    return collected_date, collected_days, weekdays, collection_weekdays


# PLOT SALES PER TIME
def plot_collection_time_of_days(collection_hours, title=""):
    hours_in_day = [i + 0.5 for i in
                    range(24)]  # Add half hours to put the value of a between hour in the middle, for now
    hours_in_day_round = [i for i in range(24)]
    nr_collection_per_hour = [np.count_nonzero(collection_hours == i) for i in range(24)]

    # Only plot hours where people actually pick up parcels
    last_nonzero = [index for index, item in enumerate(nr_collection_per_hour) if item is not 0][-1]
    first_nonzero = [index for index, item in enumerate(nr_collection_per_hour) if item is not 0][0]

    begin_hour = hours_in_day_round[first_nonzero] - 1
    end_hour = hours_in_day_round[last_nonzero] + 1

    # But always show from atleast 6 til 21
    if begin_hour > 7:
        begin_hour = 6
    if end_hour < 21:
        begin_hour = 21

    fig, ax = plt.subplots(figsize=(5, 2))
    plt.plot(hours_in_day[begin_hour:end_hour], nr_collection_per_hour[begin_hour:end_hour])
    plt.xlim(begin_hour, end_hour)
    plt.ylim(0, 1.2 * max(nr_collection_per_hour))

    # Add accompanying text
    for i in range(len(hours_in_day)):
        if nr_collection_per_hour[i] is not 0:
            plt.text(x=hours_in_day[i], y=nr_collection_per_hour[i] + 0.08 * max(nr_collection_per_hour),
                     s=nr_collection_per_hour[i], ha='center')

    # these are the plots, sets the Title and x/y labels
    plt.grid(False)
    plt.title(title)
    plt.xlabel("Time of the day")
    plt.ylabel("Number of collections")

    xtick_str = [str(int(np.linspace(begin_hour, end_hour, end_hour - begin_hour + 2)[i])) + ":00" for i in
                 range(0, len(np.linspace(begin_hour, end_hour, end_hour - begin_hour + 1)))]

    xtick_str[0] = "Before " + str(begin_hour) + ":00"
    xtick_str[-1] = "After " + str(end_hour + 1) + ":00"

    plt.xticks(hours_in_day_round[begin_hour: end_hour], xtick_str, rotation=45, ha='right')
    plt.show()
    return fig


def find_collected_not_collected_prev_x_months(company, year_nr, month_nr, nr_months_back=6):
    ym_array, ym_array_str = make_year_month_array(nr_months_back=nr_months_back)
    results_df = pd.DataFrame(columns=['ym_stamp', 'collected', 'not_collected', 'collected_pct'])
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
                                                                                             " " + str(
            add_filter_on_suppliers_strings_pup_pargo(supplier_list_pup_pargo=supplier_list_pup_pargo)) + " "

                                                                                                          "AND YEAR(orders_status.order_status_open) = " + str(
            year_month[0]) + " "
                             ";  ")

        results_old_db, column_names = send_command_to_mysql_database(command_old_db)
        parcel_counts_old_db_df = pd.DataFrame(results_old_db)
        parcel_counts_old_db_df.columns = [column_names]
        parcel_counts_old_db_df.columns = parcel_counts_old_db_df.columns.get_level_values(0)
        parcel_counts_old_db_df = parcel_counts_old_db_df.fillna(0)

        parcels_collected_old = parcel_counts_old_db_df["status_collected"][0]
        parcels_not_collected_old = parcel_counts_old_db_df["status_return_route"][0] + \
                                    parcel_counts_old_db_df["status_return_supplier"][0]

        # Now find for new database
        command_new_db = ("SELECT "
                          "SUM(CASE WHEN process_types_stages.name = 'completed' THEN 1 ELSE 0 END) AS status_collected, "
                          "SUM(CASE WHEN process_types_stages.name = 'not_completed' THEN 1 ELSE 0 END) AS status_not_collected "
                          "FROM order_processes "
                          "INNER JOIN process_types_stages ON process_types_stages.uuid = order_processes.current_process_stage "
                          "INNER JOIN suppliers ON suppliers.uuid = order_processes.owner_id "
                          "INNER JOIN process_types ON process_types.uuid = order_processes.process_type_id "
                          "WHERE "
                          "  EXTRACT (MONTH from order_processes.created_at) = " + str(year_month[1]) + " "
                                                                                                        " " + str(
            add_filter_on_suppliers_strings_simba(supplier_list_simba=supplier_list_simba)) + " "
                                                                                              "AND EXTRACT (YEAR from order_processes.created_at) = " + str(
            year_month[0]) + " "
                             "AND process_types.name  = 'w2p' ")  # Multiple process type stages from different process types (eg p2p etc) have teh same name

        results_new_db, column_names = arbitrary_command_newdatabase(command_new_db)
        parcel_counts_new_db_df = pd.DataFrame(results_new_db)
        parcel_counts_new_db_df.columns = [column_names]
        parcel_counts_new_db_df.columns = parcel_counts_new_db_df.columns.get_level_values(0)
        parcel_counts_new_db_df = parcel_counts_new_db_df.fillna(0)

        parcels_collected_new = parcel_counts_new_db_df["status_collected"][0]
        parcels_not_collected_new = parcel_counts_new_db_df["status_not_collected"][0]

        nr_parcels_collected = parcels_collected_old + parcels_collected_new
        nr_parcels_not_collected = parcels_not_collected_old + parcels_not_collected_new
        if nr_parcels_collected + nr_parcels_not_collected != 0:  # Avoid division by 0
            frac = nr_parcels_collected / (nr_parcels_collected + nr_parcels_not_collected)
        else:
            frac = 0
        results_df = results_df.append(
            {'ym_stamp': (str(ym_array_str[i][1]) + " " + str(ym_array_str[i][0])), 'collected': nr_parcels_collected,
             'not_collected': nr_parcels_not_collected, 'collected_pct': 100 * frac}, ignore_index=True)

    if [index for index, item in enumerate(results_df['collected']) if item == 0]:
        last_zero_element = [index for index, item in enumerate(results_df['collected']) if item == 0][-1]
        results_df_nonzero = results_df.iloc[last_zero_element + 1:]
        ym_array_str_nonzero = ym_array_str[last_zero_element + 1:]
        return results_df_nonzero, ym_array_str_nonzero
    else:
        return results_df, ym_array_str


# Find hours, dates and times when people pick up parcel.
def extract_col_times_hours_dates(df_times):
    collection_values = df_times.values
    collection_times = pd.to_datetime(collection_values)  # Life is better in datetime type than in npdatetime
    collection_hours = collection_times.hour + 2  # Correct for difference of time in SA from UCT
    collection_dates = collection_times.day
    return collection_hours, collection_dates, collection_times
##############################  This function takes a dataframe and extracts these values and uses the datetime library to extract values we want to plot.




################### This function plots how many people collect they parcels after work hours.
def piechart_collection_after_pre_work(collection_hours, title=" "):
    collection_after_work = np.count_nonzero(collection_hours >= 17) + np.count_nonzero(collection_hours < 9)
    collection_pre_work = len(collection_hours) - collection_after_work
    assert collection_after_work + collection_pre_work == len(collection_hours)
    fig = pie_chart(title=title, data=[collection_pre_work, collection_after_work],
                    labels=["Within office hours", "Outside of office hours"]
    return fig



def plot_collected_parcels_per_day_of_month(collection_dates, year_nr, month_nr):
    unique_day_of_col, unique_day_of_col_counts = np.unique(collection_dates, return_counts=True)
    all_days = np.linspace(1, calendar.monthrange(int(year_nr), int(month_nr))[1],
                           calendar.monthrange(int(year_nr), int(month_nr))[
                               1])  # monthrange returns nr of days in a month. Add 1 to correct for indexing
    # Make some ticks for each individual day later

    # Make sure that days with zero sales are still included
    for i in range(1, calendar.monthrange(int(year_nr), int(month_nr))[1] + 1):
        if i not in unique_day_of_col:
            unique_day_of_col_counts = np.insert(unique_day_of_col_counts, i - 1, 0)
    plt.figure(figsize=(5, 2))
    plt.plot(all_days, unique_day_of_col_counts)
    plt.title("Collections per day of the month")
    plt.ylabel("Parcels")
    plt.show()



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


# collection plots.
def plot_collection_weekdays(collection_weekdays, title):
    weekday_indices, weekday_counts = np.unique(collection_weekdays, return_counts=True)
    weekday_str_to_index = weekday_str_to_index = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                                                   4: "Friday", 5: "Saturday", 6: "Sunday"}
    weekdays = np.zeros([len(weekday_str_to_index)])
    weekdays = [weekday_str_to_index[i] for i in weekday_indices]

    y_pos = np.arange(len(weekdays))

    fig, ax = plt.subplots(figsize=(5, 2))
    plt.bar(y_pos, weekday_counts, align='center')
    plt.xticks(y_pos, weekdays, rotation=45, ha='right')  # angle in degrees
    plt.ylabel('Collections')
    plt.title(title)
    plt.ylim(0, 1.2 * max(weekday_counts))
    plt.grid(False)
    for i in range(len(weekday_counts)):
        plt.text(x=y_pos[i] - 0.2, y=weekday_counts[i] + 0.05 * max(weekday_counts), s=weekday_counts[i])
    ax.set_facecolor = 'w'
    plt.show()
    return fig


def plot_collection_monthdays(collection_monthdays, title):
    days, day_counts = np.unique(collection_monthdays, return_counts=True)

    month_days = np.arange(1, max(collection_monthdays) + 1)

    # Insert zeros if days are lacking
    for i in range(1, calendar.monthrange(int(year_nr), int(month_nr))[1] + 1):
        if i not in days:
            day_counts = np.insert(day_counts, i - 1, 0)

    y_pos = np.arange(1, calendar.monthrange(int(year_nr), int(month_nr))[1] + 1)

    fig, ax = plt.subplots(figsize=(5, 1.7))  # width, height
    plt.bar(y_pos, day_counts, align='center')
    plt.xticks(y_pos, month_days, rotation=45)  # angle in degrees
    plt.ylabel('Number of collections')
    plt.title(title)
    plt.xlabel('Day of the month')
    plt.ylim(0, 1.1 * max(day_counts))
    plt.grid(False)
    for i in range(len(day_counts)):
        plt.text(x=y_pos[i], y=day_counts[i] + 0.05 * max(day_counts), s=day_counts[i], ha='center')
    ax.set_facecolor = 'w'
    plt.show()

    return fig


def plot_collected_not_collected_df(collected_not_collected_df, title=""):
    colors = np.asarray([[255, 242, 0], [24, 120, 185], [223, 224, 224], [0, 0, 0]]) / 255
    y_pos = np.arange(len(collected_not_collected_df))
    y_min = 80

    fig, ax = plt.subplots(figsize=(5, 2))  # width, height
    plt.bar(y_pos, collected_not_collected_df["collected_pct"] - y_min, align='center', color=colors[0])
    plt.xticks(y_pos, collected_not_collected_df["ym_stamp"], rotation=45, ha='right')  # angle in degrees
    plt.ylabel('Collection rate')
    plt.title(title)
    plt.xlabel('Month')

    ax.set_ylim([0, 23])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["80", "85", "90", "95", "100"])
    plt.grid(False)
    for i in range(len(collected_not_collected_df)):
        plt.text(x=y_pos[i], y=float(collected_not_collected_df["collected_pct"].iloc[i]) - y_min + 0.01 * float(
            max(collected_not_collected_df["collected_pct"])),
                 s=str("%.2f" % float(collected_not_collected_df["collected_pct"].iloc[i])), ha='center')
    ax.set_facecolor = 'w'
    plt.show()
    return fig
