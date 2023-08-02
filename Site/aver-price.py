from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

file_path = '/home/darkhan/Личная/Dubai/Transactions.csv'
df = pd.read_csv(file_path)

# Преобразование столбца 'instance_date' в тип даты (с учетом правильного формата даты)
df['instance_date'] = pd.to_datetime(df['instance_date'], format='%d-%m-%Y')

unique_trans_group = df['trans_group_en'].unique()
unique_property_type = df['property_type_en'].unique()
unique_property_usage = df['property_usage_en'].unique()
unique_building_name = df['building_name_en'].unique()
unique_rooms = df['rooms_en'].unique()

@app.route('/', methods=['GET', 'POST'])
def index():
    submitted = False
    plot_filename = None
    selected_trans_group = ""
    selected_property_usage = ""
    selected_property_type = ""
    selected_building_name = ""
    selected_rooms = ""
    

    if request.method == 'POST':
        selected_trans_group = request.form['selected_trans_group']
        selected_property_usage = request.form['selected_property_usage']
        selected_property_type = request.form['selected_property_type']
        selected_building_name = request.form['selected_building_name']
        selected_rooms = request.form['selected_rooms']
        submitted = True

        max_price = 100000

        filtered_data = df[(df['property_usage_en'] == selected_property_usage) &
                           (df['trans_group_en'] == selected_trans_group) &
                           (df['property_type_en'] == selected_property_type) &
                           (df['building_name_en'] == selected_building_name) &
                           (df['rooms_en'] == selected_rooms) &
                           (df['meter_sale_price'] <= max_price)]

        # df['instance_date'] = pd.to_datetime(df['instance_date'], format='%d-%m-%Y')

        monthly_avg_prices = filtered_data.set_index('instance_date')['meter_sale_price'].resample('M').mean()


        plt.figure(figsize=(10, 6))
        plt.title(f'Динамика изменения средней цены на недвижимость')
        plt.xlabel('Дата сделки')
        plt.ylabel('Средняя цена')

        plt.plot(monthly_avg_prices.index.values, monthly_avg_prices.values, label='Средняя цена (по месяцам)', linestyle='-', color='b')

        # z = np.polyfit(monthly_avg_prices.index.values.astype(np.int64) // 10**9, monthly_avg_prices.values, 3)
        # p = np.poly1d(z)
        # trend_line = p(monthly_avg_prices.index.values.astype(np.int64) // 10**9)
        # plt.plot(monthly_avg_prices.index.values, trend_line, label='Полиномиальная кривая тренда', linestyle='-', color='r')

        plt.legend()

        plot_filename = 'static/plot.png'
        plt.savefig(plot_filename)

    return render_template('index.html', submitted=submitted, plot_filename=plot_filename,
                           unique_trans_group=unique_trans_group, unique_property_type=unique_property_type,
                           unique_property_usage=unique_property_usage, unique_building_name=unique_building_name,
                           unique_rooms=unique_rooms,
                           selected_trans_group=selected_trans_group,
                           selected_property_usage=selected_property_usage,
                           selected_property_type=selected_property_type,
                           selected_building_name=selected_building_name,
                           selected_rooms=selected_rooms)

if __name__ == '__main__':
    app.run(debug=True)
