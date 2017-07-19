
import pandas as pd
from sklearn.cluster import KMeans

location = '/home/svc/abcde/instacart data/'

PriorProductOrders = pd.read_csv(location + 'order_products__prior.csv')
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

user_aggregate = grouped.agg({'department_id': lambda x: list(x),
                              'order_hour_of_day': lambda x: list(x),
                              'days_since_prior_order': lambda x: list(x),
                              'order_dow': lambda x: list(x),
                              'reordered': lambda x: list(x)
                              })
del grouped

df = pd.DataFrame(data=None, columns=['user_id'])
df['user_id'] = user_aggregate['user_id']

for i in range(1, 22):
    df['department_'+str(i)] = user_aggregate['department_id'].apply(lambda x: x.count(i))

print('', df.head(20).to_string())
df['morning_order'] = user_aggregate['order_hour_of_day'].apply(lambda x: x.count(1))
df['afternoon_order'] = user_aggregate['order_hour_of_day'].apply(lambda x: x.count(2))
df['night_order'] = user_aggregate['order_hour_of_day'].apply(lambda x: x.count(3))
df['week_day_order'] = user_aggregate['order_dow'].apply(lambda x: x.count(1))
df['week_end_order'] = user_aggregate['order_dow'].apply(lambda x: x.count(0))
df['prior_order_days_0_10'] = user_aggregate['days_since_prior_order'].apply(lambda x: x.count(1))
df['prior_order_days_10_20'] = user_aggregate['days_since_prior_order'].apply(lambda x: x.count(2))
df['prior_order_days_20_30'] = user_aggregate['days_since_prior_order'].apply(lambda x: x.count(3))
df['prior_order_days_30+'] = user_aggregate['days_since_prior_order'].apply(lambda x: x.count(4))
df['re_order'] = user_aggregate['reordered'].apply(lambda x: x.count(1))
df['first_time_order'] = user_aggregate['reordered'].apply(lambda x: x.count(0))

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
