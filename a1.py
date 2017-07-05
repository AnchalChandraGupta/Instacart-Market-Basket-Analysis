
import pandas as pd
from sklearn.cluster import KMeans
import csv

location = '/home/anchal/Desktop/instacart data/'

order_products_prior = pd.read_csv(location + 'order_products__prior.csv', usecols= ['order_id', 'product_id'])
orders = pd.read_csv(location + 'orders.csv', usecols=['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])
orders['days_since_prior_order'] = orders['days_since_prior_order'].apply(lambda x: 1 if 0 <= x <= 10 else 2 if 10 < x <= 20 else 3 if 20 < x <= 30 else 4)
orders['order_hour_of_day'] = orders['order_hour_of_day'].apply(lambda x: 1 if x < 12 else 2 if x < 17 else 3 if x < 24 else 0)
orders['order_dow'] = orders['order_dow'].apply(lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else 0 if(x == 5 or x == 6) else 2)

joined = pd.merge(orders, order_products_prior)
del order_products_prior
del orders


products = pd.read_csv(location + 'products.csv', usecols=['product_id', 'department_id'])
joined = pd.merge(joined, products)
del products

departments = pd.read_csv(location + 'departments.csv', usecols=['department_id'])
joined = pd.merge(joined, departments)
del departments

grouped = joined.groupby('user_id', as_index=False)
del joined

def to_set(x):
    return set(x)

user_aggregate = grouped.agg({'department_id': to_set,\
                      'order_hour_of_day': to_set,\
                      'days_since_prior_order': to_set,\
                      'order_dow': to_set})
user_aggregate.columns = ['user_id', 'department_set', 'hour_of_day_set', 'days_since_prior_order_set', 'day_of_week_set']
del grouped


print(user_aggregate.head(20).to_string())


df = pd.DataFrame(data=None, columns=['user_id'])
df['user_id'] = user_aggregate['user_id']

for i in range(1, 22):
    df['department_'+str(i)] = user_aggregate['department_set'].apply(lambda x: True if i in x else False)
del user_aggregate['department_set']
df['morning_order'] = user_aggregate['hour_of_day_set'].apply(lambda x: True if 1 in x else False)
df['afternoon_order'] = user_aggregate['hour_of_day_set'].apply(lambda x: True if 2 in x else False)
df['night_order'] = user_aggregate['hour_of_day_set'].apply(lambda x: True if 3 in x else False)
del user_aggregate['hour_of_day_set']
df['week_day_order'] = user_aggregate['day_of_week_set'].apply(lambda x: True if 1 in x else False)
df['week_end_order'] = user_aggregate['day_of_week_set'].apply(lambda x: True if 0 in x else False)
del user_aggregate['day_of_week_set']
df['prior_order_days_0_10'] = user_aggregate['days_since_prior_order_set'].apply(lambda x: True if 1 in x else False)
df['prior_order_days_10_20'] = user_aggregate['days_since_prior_order_set'].apply(lambda x: True if 2 in x else False)
df['prior_order_days_20_30'] = user_aggregate['days_since_prior_order_set'].apply(lambda x: True if 3 in x else False)
df['prior_order_days_30+'] = user_aggregate['days_since_prior_order_set'].apply(lambda x: True if 4 in x else False)

del user_aggregate

print('df','\n', df.head(100).to_string(), '\n\n')
df.to_csv(location+'df.csv', index=False)
print(df.shape)

kmeans = KMeans(n_clusters=20).fit(df)
print(kmeans.labels_)


write_to_file = open(location+'kmeans_labels_1.csv', 'w')

for i in range(len(df)):
    write_to_file.write(df(i))
    write_to_file.write('    ')
    write_to_file.write(str(kmeans.labels_[i]))
    write_to_file.write('\n')
