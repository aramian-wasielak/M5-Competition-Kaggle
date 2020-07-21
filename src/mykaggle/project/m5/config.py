from pathlib import Path

input_dir = Path('../data/input')
output_dir = Path('../data/output')

target_col = 'units_sold'
pred_target_col = 'pred_' + target_col
pred_date_col = 'pred_date'

train_start_date = '2016-02-01'
train_end_date = '2016-05-15'

test_start_date = '2016-05-16'
test_end_date = '2016-05-22'

store_list = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

pred_week_list = [1, 2, 3, 4]
