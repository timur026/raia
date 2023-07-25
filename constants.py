# List of unwanted categories
unwanted_categories = [
    "Adding Land By Sell",
    "Delayed Lease to Own Modify",
    "Lease Development Modify",
    "Transfer Development Mortgage",
    "Modify Delayed Mortgage",
    "Lease to Own on Development Modification",
    "Development Mortgage Pre-Registration",
    "Modify Mortgage Pre-Registration",
    "Mortgage Transfer Pre-Registration",
    "Lease Finance Modification",
    "Delayed Sell Development",
    "Sell Development - Pre Registration",
    "Delayed Sell Lease to Own Registration",
    "Lease to Own Modify",
    "Modify Development Mortgage",
    "Lease to Own on Development Registration",
    "Lease to Own Transfer",
    "Sale On Payment Plan",
    "Delayed Lease to Own Registration",
    "Lease Development Registration",
    "Delayed Development",
    "Mortgage Transfer",
    "Modify Delayed Mortgage ",
    "Development Registration Pre-Registration",
    "Development Mortgage",
    "Lease Finance Registration",
    "Modify Mortgage"
]

# Create a list of columns to be removed
columns_to_remove = ['trans_group_ar', 'procedure_name_ar', 'property_type_ar',
                     'property_sub_type_ar', 'property_usage_ar', 'reg_type_ar',
                     'area_name_ar', 'building_name_ar', 'project_name_ar',
                     'master_project_ar', 'nearest_landmark_ar', 'nearest_metro_ar',
                     'nearest_mall_ar', 'rooms_ar',
                     'procedure_id', 'trans_group_id', 'property_type_id', 'property_sub_type_id',
                     'reg_type_id', 'area_id', 'rent_value', 'meter_rent_price', 'no_of_parties_role_1',
                     'no_of_parties_role_2', 'no_of_parties_role_3',
                     'project_number', 'meter_sale_price']

area_limits = [25, 500]
price_limits = [65000, 6500000]
