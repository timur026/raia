import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import constants


def delete_arabic_and_indices(source, destination):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Drop the identified columns from the DataFrame
    df.drop(columns=constants.columns_to_remove, inplace=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(destination, index=False)


def has_parking_to_yes_no(source, destination):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Replace values in the "has_parking" column
    df['has_parking'].replace({0: 'No', 1: 'Yes'}, inplace=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(destination, index=False)


def analyse_data(source):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # For information purpose only
    print("Length:", len(df))
    print("Number of columns:", len(df.columns))

    # Count the number of empty entries (NaN or None) in each column
    empty_counts = df.isnull().sum()

    # Print the number of empty entries in each column
    print("Number of empty entries in each column:")
    print(empty_counts)


def leave_residential_only(source, destination):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Filter the DataFrame to keep only rows with "Residential" in the "property_usage_en" column
    df_residential = df[df['property_usage_en'] == 'Residential']

    # Save the filtered DataFrame back to the CSV file
    df_residential.to_csv(destination, index=False)


def delete_from_procedure_name_en(source, destination):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Filter the DataFrame to exclude rows with unwanted categories in the "procedure_name_en" column
    df_filtered = df[~df['procedure_name_en'].isin(constants.unwanted_categories)].copy()

    # Save the filtered DataFrame back to the CSV file
    df_filtered.to_csv(destination, index=False)


def get_column_stats(source, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Calculate statistics for the "procedure_area" column
    column_stats = df[column_name].describe()

    # Print the statistics
    print("Statistics for the {} column:".format(column_name))
    print(column_stats)


def drop_empty_values_in_column(source, destination, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Drop rows with empty data in the "nearest_metro_en" column
    data = df.dropna(subset=[column_name])

    # Write the modified DataFrame back to a new CSV file
    data.to_csv(destination, index=False)


def column_statistics(source, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Calculate statistics
    column_max = df[column_name].max()
    column_min = df[column_name].min()
    column_mean = df[column_name].mean()
    column_median = df[column_name].median()
    column_std = df[column_name].std()

    # Print the statistics with two decimal places
    print("Maximum: {:,.2f}".format(column_max))
    print("Minimum: {:,.2f}".format(column_min))
    print("Mean: {:,.2f}".format(column_mean))
    print("Median: {:,.2f}".format(column_median))
    print("Standard Deviation: {:,.2f}".format(column_std))

    print("Range: ({:,.2f}, {:,.2f})".format(column_mean - 1 * column_std, column_mean + 1 * column_std))


def print_columns_and_categories(source, unique_categories_number):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Get the data types of each column in the DataFrame
    data_types = df.dtypes

    # Filter the columns that are non-numeric (categorical columns)
    categorical_columns = data_types[data_types == 'object'].index

    # Print the names of categories for columns with less than 20 unique categories
    print("Categories for columns with less than 20 unique categories:")
    for column in categorical_columns:
        num_categories = df[column].nunique()
        if num_categories < unique_categories_number:
            print(f"{column}: {df[column].unique()}")
            print("\n")


def delete_one_column(source, destination, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Drop the identified columns from the DataFrame
    df.drop(columns=[column_name], inplace=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(destination, index=False)


def delete_nan_from_column(source, destination, column_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Drop rows with NaN values in the "rooms_en" column
    df.dropna(subset=[column_name], inplace=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(destination, index=False)


def delete_categories_from_column(source, destination, column_name, categories_to_delete):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Filter the DataFrame to exclude rows with the specified categories in 'rooms_en'
    df_filtered = df[~df[column_name].isin(categories_to_delete)]

    # Save the filtered DataFrame back to the CSV file
    df_filtered.to_csv(destination, index=False)


def analyse_column(source, column):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Get the value counts of each category in the "property_sub_type_en" column
    column_counts = df[column].value_counts(dropna=False)

    # Print the categories and their occurrences, including the number of empty occurrences
    print("Categories of {} with number of occurrences:".format(column))
    print(column_counts)


def trim_outliers(source, destination, column_name, lower_bound, upper_bound):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(source)

    # Filter the DataFrame to keep only rows with "actual_worth" values within the 2nd and 98th percentiles
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Save the filtered DataFrame back to the CSV file
    df_filtered.to_csv(destination, index=False)


def trim_by_date(source, destination, start_date, end_date):
    # Read the CSV file into a pandas DataFrame, specify the date format
    df = pd.read_csv(source, parse_dates=['instance_date'], dayfirst=True)

    # Filter the DataFrame to keep only rows with a date greater than or equal to "01/01/2022"
    df_filtered = df[(df['instance_date'] >= start_date) & (df['instance_date'] <= end_date)]

    # Save the filtered DataFrame back to the CSV file
    df_filtered.to_csv(destination, index=False)


if __name__ == '__main__':
    master_file = r"C:\Users\Timur\Desktop\Dubai_rev2\Transactions_MASTER.csv"
    source_file = r"C:\Users\Timur\Desktop\Dubai_rev2\Transactions.csv"
    destination_file = r"C:\Users\Timur\Desktop\Dubai_rev2\Transactions.csv"

    # get_column_stats(source_file, "actual_worth")
    # get_column_stats(source_file, "procedure_area")
    # column_statistics(source_file, "procedure_area")
    # print_columns_and_categories(source_file, 1000)
    # analyse_column(source_file, "property_sub_type_en")
    # analyse_column(source_file, "reg_type_en")

    # delete_arabic_and_indices(master_file, destination_file)  #  DONE
    # has_parking_to_yes_no(source_file, destination_file)  #  DONE
    # leave_residential_only(source_file, destination_file)  #  DONE
    # delete_from_procedure_name_en(source_file, destination_file)  #  DONE
    # drop_empty_values_in_column(source_file, destination_file, "actual_worth")  #  DONE
    # delete_one_column(source_file, destination_file, "property_type_en")  #  DONE
    # delete_nan_from_column(source_file, destination_file, "rooms_en")  #  DONE
    # categories = ["Gifts"]
    # delete_categories_from_column(source_file, destination_file, "trans_group_en", categories)  #  DONE
    # categories = ["Stacked Townhouses"]
    # delete_categories_from_column(source_file, destination_file, "property_sub_type_en", categories)  #  DONE
    # trim_outliers(source_file, destination_file, "actual_worth", constants.price_limits[0], constants.price_limits[1])
    # trim_by_date(source_file, destination_file, '01/01/2022', '25/07/2023')
    analyse_data(destination_file)



