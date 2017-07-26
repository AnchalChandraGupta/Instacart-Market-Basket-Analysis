import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

pd.set_option('max_colwidth', 200)

location = '/home/anchal/PycharmProjects/instacart_data/'

prior_product_orders = pd.read_csv(location + 'order_products__prior.csv', nrows=100000)
train_product_orders = pd.read_csv(location + 'order_products__train.csv')

user_orders = pd.read_csv(location + 'orders.csv', usecols=['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])
user_orders['days_since_prior_order'] = user_orders['days_since_prior_order'].apply(lambda x: 1 if 0 <= x <= 10 else 2 if 10 < x <= 20 else 3 if 20 < x <= 30 else 4)
user_orders['order_hour_of_day'] = user_orders['order_hour_of_day'].apply(lambda x: 1 if x < 12 else 2 if x < 17 else 3 if x < 24 else 0)
user_orders['order_dow'] = user_orders['order_dow'].apply(lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else 0 if(x == 5 or x == 6) else 2)

joined = pd.merge(user_orders, prior_product_orders)
del prior_product_orders
del user_orders

products = pd.read_csv(location + 'products.csv', usecols=['product_id', 'department_id'])
joined = pd.merge(joined, products)
del products

departments = pd.read_csv(location + 'departments.csv', usecols=['department_id'])
joined = pd.merge(joined, departments)
del departments

grouped = joined.groupby('user_id', as_index=False)

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
    df['department_c_'+str(i)] = user_aggregate['department_id'].apply(lambda x: x.count(i))
for i in [1,2,3]:
    df['hour_of_day_c_'+str(i)] = user_aggregate['order_hour_of_day'].apply(lambda x: x.count(i))
for i in [1,2]:
    df['day_of_week_c_'+str(i)] = user_aggregate['order_dow'].apply(lambda x: x.count(i))
for i in [1, 2, 3, 4]:
    df['prior_order_days_c_'+str(i)] = user_aggregate['days_since_prior_order'].apply(lambda x: x.count(i))
for i in [1,2]:
    df['order_reorder_c_'+str(i)] = user_aggregate['reordered'].apply(lambda x: x.count(1))

del user_aggregate

print('df','\n', df.head(100).to_string(), '\n\n')
df.to_csv(location+'df1.csv', index=False)
print(df.shape)

matrix = df.drop(['user_id'], axis=1)
print('1')

kmeans = KMeans(n_clusters=200, random_state=1).fit(matrix)
print('1')

write_to_file = open(location+'kmeans_labels_.csv', 'w')

user_label_list = zip(df.user_id, kmeans.labels_)
df['cluster_id'] = pd.DataFrame(kmeans.labels_)
joined = pd.merge(joined, df[['user_id', 'cluster_id']])

print('\n\n\n\njoined\n\n', joined.head(200).to_string())

product_aggregate = joined.groupby(by=['cluster_id'], as_index=False).agg({'product_id': lambda x: list(x)})
print(product_aggregate.head(200).to_string())

product_aggregate['product_id'] = product_aggregate['product_id'].apply(lambda x: Counter(w for w in x).most_common(10))
print(product_aggregate.head(200).to_string())

print('1')
for i in user_label_list:
    write_to_file.write('    ')
    write_to_file.write(str(i))
    write_to_file.write('\n')

write_to_file = open(location+'kmeans_labels_1.csv', 'w')

print('0')
