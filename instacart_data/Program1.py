import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import random
from sklearn import svm

pd.set_option('max_colwidth', 220)

location = '/home/anchal/PycharmProjects/Instacart-Market-Basket-Analysis/instacart_data/'


print('\n\n\n\naisles\n\n' + pd.read_csv(location+'aisles.csv', nrows=10).head(10).to_string())
print('\n\n\n\ndepartments\n\n' + pd.read_csv(location+'departments.csv', nrows=10).head(10).to_string())
print('\n\n\n\norder_products__prior\n\n' + pd.read_csv(location+'order_products__prior.csv', nrows=10).head(10).to_string())
print('\n\n\n\norder_products__train\n\n' + pd.read_csv(location+'order_products__train.csv', nrows=10).head(10).to_string())
print('\n\n\n\norders\n\n' + pd.read_csv(location+'orders.csv', nrows=10).head(10).to_string())
print('\n\n\n\nproducts\n\n' + pd.read_csv(location+'products.csv', nrows=10).head(10).to_string())

prior_product_orders = pd.read_csv(location + 'order_products__prior.csv')
train_product_orders = pd.read_csv(location + 'order_products__train.csv')

user_orders = pd.read_csv(location + 'orders.csv', usecols=['order_id', 'user_id', 'eval_set', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])
user_orders['days_since_prior_order'] = user_orders['days_since_prior_order'].apply(lambda x: 1 if 0 <= x <= 10 else 2 if 10 < x <= 20 else 3 if 20 < x <= 30 else 4)
user_orders['order_hour_of_day'] = user_orders['order_hour_of_day'].apply(lambda x: 1 if x < 12 else 2 if x < 17 else 3 if x < 24 else 0)
user_orders['order_dow'] = user_orders['order_dow'].apply(lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else 0 if(x == 5 or x == 6) else 2)

train_user_id_set = set()
test_user_id_set = set()
count=0

for i in user_orders[['user_id', 'eval_set']].iterrows():
    count+=1
    if count%100000 is 0:
        print(count/100000)
    if str(i[1]['eval_set']).__contains__('train'):
        train_user_id_set.add(i[1]['user_id'])
    if str(i[1]['eval_set']).__contains__('test'):
        test_user_id_set.add(i[1]['user_id'])

print('\n\n\n\ntrain_user_id_set length', len(train_user_id_set))
print('\n\n\n\ntest_user_id_set length', len(test_user_id_set))


training_set = random.random(train_user_id_set,100000)
validation
joined = pd.merge(user_orders, prior_product_orders)
del prior_product_orders

products = pd.read_csv(location + 'products.csv', usecols=['product_id', 'department_id'])
joined = pd.merge(joined, products)
del products

departments = pd.read_csv(location + 'departments.csv', usecols=['department_id'])
joined = pd.merge(joined, departments)
print('\n\n\n\njoined    '+str(joined.shape)+'\n\n',joined.head(10).to_string())
del departments

# train_user_id_set: set contain the id of the training users


training_user_join = joined.loc[joined['user_id'].isin(train_user_id_set)]
test_user_join = joined.loc[joined['user_id'].isin(test_user_id_set)]
del joined

print('\n\n\n\ntraining_user_join\n\n', training_user_join.head(10).to_string())
print('\n\n\n\ntest_user_join\n\n', test_user_join.head(10).to_string())



# training users

train_user_grouped = training_user_join.groupby(['user_id'], as_index=False)
train_user_dataframe = train_user_grouped.agg({'department_id': lambda x: list(x),
                              'order_hour_of_day': lambda x: list(x),
                              'days_since_prior_order': lambda x: list(x),
                              'order_dow': lambda x: list(x),
                              'reordered': lambda x: list(x)
                                               })

reduced_data_matrix = pd.DataFrame(data=None, columns=['user_id'])
reduced_data_matrix['user_id'] = train_user_dataframe['user_id']

for i in range(1, 22):
    reduced_data_matrix['department_c_' + str(i)] = train_user_dataframe['department_id'].apply(lambda x: x.count(i))
for i in [1,2,3]:
    reduced_data_matrix['hour_of_day_c_' + str(i)] = train_user_dataframe['order_hour_of_day'].apply(lambda x: x.count(i))
for i in [1,2]:
    reduced_data_matrix['day_of_week_c_' + str(i)] = train_user_dataframe['order_dow'].apply(lambda x: x.count(i))
for i in [1, 2, 3, 4]:
    reduced_data_matrix['prior_order_days_c_' + str(i)] = train_user_dataframe['days_since_prior_order'].apply(lambda x: x.count(i))
for i in [1,2]:
    reduced_data_matrix['order_reorder_c_' + str(i)] = train_user_dataframe['reordered'].apply(lambda x: x.count(1))
del train_user_dataframe

# feature matrix - each row contain feature values for 1 user       total length 131209     Number of columns (features) 32
matrix = reduced_data_matrix.drop(['user_id'], axis=1)
print('\n\n\n\nfeatures matrix\n\n', matrix.head(10).to_string(), '\n\n')
print('\n\n\n\nfeature matrix dimension ', matrix.shape)




# Kmeans fit()

kmeans = KMeans(n_clusters=200, random_state=1).fit(matrix)

# # write user_id, cluster_id to csv file
# user_cluster_list = zip(reduced_data_matrix.user_id, kmeans.labels_)
# user_cluster_list.to_csv(location + 'user_cluster_list.csv', index=False)
print('\n\n\n\n kmeans labels length\n', len(kmeans.labels_))


# get all products per per cluster
#add cluster values to joined table
reduced_data_matrix['cluster_id'] = pd.DataFrame(kmeans.labels_)
training_user_join = pd.merge(training_user_join, reduced_data_matrix[['user_id', 'cluster_id']])
print('\n\n\n\ntraining_user_join\n\n', training_user_join.head(10).to_string())

# create a list of products in each cluster using prior orders made by the users in that cluster
cluster_product_aggregate = training_user_join.groupby(by=['cluster_id'], as_index=False).agg({'product_id': lambda x: list(x)})
print('\n\n\n\ncluster_product_aggregate\n\n', cluster_product_aggregate.head(10).to_string())

# get the top 10 products from each cluster
cluster_product_aggregate['product_id'] = cluster_product_aggregate['product_id'].apply(lambda x: Counter(w for w in x).most_common(10))
print('\n\n\n\ncluster_product_aggregate top 10 products for each cluster\n\n', cluster_product_aggregate.head(10).to_string())




# test users

test_user_grouped = test_user_join.groupby(['user_id'], as_index=False)
test_user_dataframe = test_user_grouped.agg({'department_id': lambda x: list(x),
                              'order_hour_of_day': lambda x: list(x),
                              'days_since_prior_order': lambda x: list(x),
                              'order_dow': lambda x: list(x),
                              'reordered': lambda x: list(x)
                                               })

reduced_test_data_matrix = pd.DataFrame(data=None, columns=['user_id'])
reduced_test_data_matrix['user_id'] = test_user_dataframe['user_id']
for i in range(1, 22):
    reduced_test_data_matrix['department_c_' + str(i)] = test_user_dataframe['department_id'].apply(lambda x: x.count(i))
for i in [1,2,3]:
    reduced_test_data_matrix['hour_of_day_c_' + str(i)] = test_user_dataframe['order_hour_of_day'].apply(lambda x: x.count(i))
for i in [1,2]:
    reduced_test_data_matrix['day_of_week_c_' + str(i)] = test_user_dataframe['order_dow'].apply(lambda x: x.count(i))
for i in [1, 2, 3, 4]:
    reduced_test_data_matrix['prior_order_days_c_' + str(i)] = test_user_dataframe['days_since_prior_order'].apply(lambda x: x.count(i))
for i in [1,2]:
    reduced_test_data_matrix['order_reorder_c_' + str(i)] = test_user_dataframe['reordered'].apply(lambda x: x.count(1))
del test_user_dataframe

# test feature matrix - each row contain feature values for 1 user      total length 75000      Number of columns (features) 32
test_matrix = reduced_test_data_matrix.drop(['user_id'], axis=1)
print('\n\n\n\ntest feature matrix\n\n', test_matrix.head(10).to_string(), '\n\n')
print('\n\n\n\ntest feature matrix dimension:', test_matrix.shape)




# Kmean predict()
# predict cluster for each test user

kmeans.predict(test_matrix)

test_users_predicted_clusters = kmeans.labels_
print('\n\n\n\n kmeans labels length\n', len(kmeans.labels_))
print('\n\n\n\ntest_users_predicted_clusters\n\n', test_users_predicted_clusters)
# predicted_user_label_list = zip(reduced_data_matrix.user_id, kmeans.labels_)
# predicted_user_label_list.to_csv(location + 'predicted_user_label_list.csv', index=False)

reduced_test_data_matrix['cluster_id'] = pd.DataFrame(kmeans.labels_)
a = pd.merge(cluster_product_aggregate, reduced_test_data_matrix, on='cluster_id')
user_orders = pd.read_csv(location + 'orders.csv', usecols=['order_id', 'user_id', 'eval_set'])
a = pd.merge(a, user_orders)
a = a[a['eval_set']=='test']
a = a[['order_id', 'product_id']]


# cluster the output of the training set into n (current value = 50) cluster to to get n new classes to classify the training users and test users, then use svc to classify the test users.

training_product_orders_join = pd.merge(user_orders, train_product_orders)
print('\ntraining_product_orders_join shape ', training_product_orders_join.shape)
products = pd.read_csv(location + 'products.csv', usecols=['product_id', 'aisle_id', 'department_id'])
training_product_orders_join = pd.merge(training_product_orders_join, products)

del products, train_product_orders,reduced_data_matrix

training_product_orders_grouped = training_product_orders_join.groupby(['user_id'], as_index=False)

user_aggregate1 = training_product_orders_grouped.agg({'aisle_id': lambda x: list(x),
                                                       'department_id': lambda x: list(x)})


# training output product data matrix to cluster the training output in n(current value=50) cluster(output classes)
training_set_output_df_matrix = pd.DataFrame()
training_set_output_df_matrix['user_id'] = user_aggregate1['user_id']
for i in range(1, 135):
    training_set_output_df_matrix['aisle_c_' + str(i)] = user_aggregate1['aisle_id'].apply(lambda x: x.count(i))
for i in range(1, 22):
    training_set_output_df_matrix['department_c_' + str(i)] = user_aggregate1['department_id'].apply(lambda x: x.count(i))
class_matrix = training_set_output_df_matrix.drop(['user_id'], axis=1)
print('class_matrix', class_matrix.shape)



# kmeans on training output product data matrix
training_output_kmeans = KMeans(n_clusters=50, random_state=1).fit(class_matrix)

# user_classification_list = zip(training_set_output_df_matrix.user_id, training_output_kmeans.labels_)
# user_classification_list.to_csv(location + 'user_classification_list.csv', index=False)

training_set_output_df_matrix['cluster_id'] = pd.DataFrame(training_output_kmeans.labels_)
print(5)

training_product_orders_join = pd.merge(training_product_orders_join, training_set_output_df_matrix[['user_id', 'cluster_id']])
print(6)
