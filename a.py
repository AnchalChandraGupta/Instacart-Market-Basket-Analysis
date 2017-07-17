
import pandas as pd
from sklearn.cluster import KMeans

location = '/home/svc/abcde/instacart data/'

PriorProductOrders = pd.read_csv(location + 'order_products__prior.csv', nrows=10000)
TrainProductOrders = pd.read_csv(location + 'order_products__train.csv')['order_id']
print(PriorProductOrders.head(100).to_string())
print(PriorProductOrders.shape)
UserOrders = pd.read_csv(location + 'orders.csv', usecols=['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])
UserOrders['days_since_prior_order'] = UserOrders['days_since_prior_order'].apply(lambda x: 1 if 0 <= x <= 10 else 2 if 10 < x <= 20 else 3 if 20 < x <= 30 else 4)
UserOrders['order_hour_of_day'] = UserOrders['order_hour_of_day'].apply(lambda x: 1 if x < 12 else 2 if x < 17 else 3 if x < 24 else 0)
UserOrders['order_dow'] = UserOrders['order_dow'].apply(lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else 0 if(x == 5 or x == 6) else 2)

joined = pd.merge(UserOrders, PriorProductOrders)
del PriorProductOrders
del UserOrders

products = pd.read_csv(location + 'products.csv', usecols=['product_id', 'department_id'])
joined = pd.merge(joined, products)
del products

departments = pd.read_csv(location + 'departments.csv', usecols=['department_id'])
joined = pd.merge(joined, departments)
del departments

grouped = joined.groupby('user_id', as_index=False)

del joined

def to_dict(x):
    return x.value_counts().to_dict()

user_aggregate = grouped.agg({'department_id': to_dict,
                              'order_hour_of_day': to_dict,
                              'days_since_prior_order': to_dict,
                              'order_dow': to_dict,
                              'reordered':to_dict
                              })

user_aggregate.columns = ['user_id',
                          'department_count_dict',
                          'hour_of_day_count_dict',
                          'days_prior_order_count_dict',
                          'day_of_week_count_dict',
                          'reordered_count_dict'
                          ]

del grouped

df = pd.DataFrame(data=None, columns=['user_id'])
df['user_id'] = user_aggregate['user_id']

for i in range(1, 22):
    df['department_'+str(i)] = user_aggregate['department_count_dict'].apply(lambda x: x.get(i) if x.__contains__(i) else 0)

print('', df.head(20).to_string())
del user_aggregate['department_count_dict']
df['morning_order'] = user_aggregate['hour_of_day_count_dict'].apply(lambda x: x.get(1) if x.__contains__(1) else 0)
df['afternoon_order'] = user_aggregate['hour_of_day_count_dict'].apply(lambda x: x.get(2) if x.__contains__(2) else 0)
df['night_order'] = user_aggregate['hour_of_day_count_dict'].apply(lambda x: x.get(3) if x.__contains__(3) else 0)
del user_aggregate['hour_of_day_count_dict']
df['week_day_order'] = user_aggregate['day_of_week_count_dict'].apply(lambda x: x.get(1) if x.__contains__(1) else 0)
df['week_end_order'] = user_aggregate['day_of_week_count_dict'].apply(lambda x: x.get(0) if x.__contains__(0) else 0)
del user_aggregate['day_of_week_count_dict']
df['prior_order_days_0_10'] = user_aggregate['days_prior_order_count_dict'].apply(lambda x: x.get(1) if x.__contains__(1) else 0)
df['prior_order_days_10_20'] = user_aggregate['days_prior_order_count_dict'].apply(lambda x: x.get(2) if x.__contains__(2) else 0)
df['prior_order_days_20_30'] = user_aggregate['days_prior_order_count_dict'].apply(lambda x: x.get(3) if x.__contains__(3) else 0)
df['prior_order_days_30+'] = user_aggregate['days_prior_order_count_dict'].apply(lambda x: x.get(4) if x.__contains__(4) else 0)
del user_aggregate['days_prior_order_count_dict']
df['re_order'] = user_aggregate['reordered_count_dict'].apply(lambda x: x.get(1) if x.__contains__(1) else 0)
df['first_time_order'] = user_aggregate['reordered_count_dict'].apply(lambda x: x.get(0) if x.__contains__(0) else 0)

del user_aggregate

print('df','\n', df.head(100).to_string(), '\n\n')
df.to_csv(location+'df1.csv', index=False)
print(df.shape)

kmeans = KMeans(n_clusters=20).fit(df)

write_to_file = open(location+'kmeans_labels_.csv', 'w')

for i in range(len(df)):
    write_to_file.write('    ')
    write_to_file.write(str(kmeans.labels_[i]))
    write_to_file.write('\n')

write_to_file = open(location+'kmeans_labels_1.csv', 'w')

print('10')
