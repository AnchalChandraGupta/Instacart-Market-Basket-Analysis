import pandas
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import svm
from sklearn.model_selection import train_test_split


pandas.set_option('max_colwidth', 220)
instacart_data_location = '/home/anchal/PycharmProjects/Instacart-Market-Basket-Analysis/instacart_data/'


# print 10 records from each csv file

print('\n\n\n\naisles\n\n' + pandas.read_csv(instacart_data_location + 'aisles.csv', nrows=20).head(20).to_string())
print('\n\n\n\ndepartments\n\n' + pandas.read_csv(instacart_data_location + 'departments.csv', nrows=20).head(20).to_string())
print('\n\n\n\norder_products__prior\n\n' + pandas.read_csv(instacart_data_location + 'order_products__prior.csv', nrows=20).head(20).to_string())
print('\n\n\n\norder_products__train\n\n' + pandas.read_csv(instacart_data_location + 'order_products__train.csv', nrows=20).head(20).to_string())
print('\n\n\n\norders\n\n' + pandas.read_csv(instacart_data_location + 'orders.csv', nrows=20).head(20).to_string())
print('\n\n\n\nproducts\n\n' + pandas.read_csv(instacart_data_location + 'products.csv', nrows=20).head(20).to_string())

prior_order_products = pandas.read_csv(instacart_data_location + 'order_products__prior.csv')
training_order_products = pandas.read_csv(instacart_data_location + 'order_products__train.csv')

orders = pandas.read_csv(instacart_data_location + 'orders.csv')
orders['days_since_prior_order'] = orders['days_since_prior_order'].apply(lambda x: 1 if 0 <= x <= 10 else 2 if 10 < x <= 20 else 3 if 20 < x <= 30 else 4)
orders['order_hour_of_day'] = orders['order_hour_of_day'].apply(lambda x: 1 if x < 12 else 2 if x < 17 else 3 if x < 24 else 0)
orders['order_dow'] = orders['order_dow'].apply(lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else 0 if(x == 5 or x == 6) else 2)


# train_user_id_set: set: contains the user_id of the training users
# test_user_id_set: set: contains the user_id of the test users

train_user_id_set = set()
test_user_id_set = set()
count=0
for i in orders[['user_id', 'eval_set']].iterrows():
    count+=1
    if count%100000 is 0:
        print(count/100000)
    if str(i[1]['eval_set']).__contains__('train'):
        train_user_id_set.add(i[1]['user_id'])
    if str(i[1]['eval_set']).__contains__('test'):
        test_user_id_set.add(i[1]['user_id'])
print('\n\n\n\ntrain_user_id_set length', len(train_user_id_set))
print('\n\n\n\ntest_user_id_set length', len(test_user_id_set))


# joined is join of records in tables from orders, prior_order_products, products, departments.

joined = pandas.merge(orders, prior_order_products)
del prior_order_products
products = pandas.read_csv(instacart_data_location + 'products.csv', usecols=['product_id', 'department_id'])
joined = pandas.merge(joined, products)
departments = pandas.read_csv(instacart_data_location + 'departments.csv', usecols=['department_id'])
joined = pandas.merge(joined, departments)
del products, departments
print('\n\n\n\njoined    '+str(joined.shape)+'\n\n',joined.head(20).to_string())


# training_user_join: dataframe: contains records for training users
# test_user_join: dataframe: contains records for test users

training_user_join = joined.loc[joined['user_id'].isin(train_user_id_set)]
test_user_join = joined.loc[joined['user_id'].isin(test_user_id_set)]
del joined

# training users
# group by user_id and aggregate the attributes

train_user_grouped = training_user_join.groupby(['user_id'], as_index=False)
train_user_dataframe = train_user_grouped.agg({'department_id': lambda x: list(x),
                              'order_hour_of_day': lambda x: list(x),
                              'days_since_prior_order': lambda x: list(x),
                              'order_dow': lambda x: list(x),
                              'reordered': lambda x: list(x)
                                               })


# reduced_data_matrix: Dataframe: feature matrix for training users, each row has a user_id and 32 feature values

reduced_data_matrix = pandas.DataFrame(data=None, columns=['user_id'])
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


# X_training_reduced_data_matrix: Dataframe: training feature matrix
# X_validation_reduced_data_matrix: Dataframe: validation feature matrix

X_training_reduced_data_matrix, X_validation_reduced_data_matrix = train_test_split(reduced_data_matrix, test_size=31209)
del reduced_data_matrix
print('\n\n\n\nX_training_reduced_data_matrix\n\n', X_training_reduced_data_matrix.shape)
print('\n\n\n\nX_validation_reduced_data_matrix\n\n', X_validation_reduced_data_matrix.shape)


# matrix: Dataframe: each row contain 32 feature values for a user

matrix = X_training_reduced_data_matrix.drop(['user_id'], axis=1)
print('\n\n\n\nfeatures matrix\n\n', matrix.head(20).to_string(),
      '\n\nfeature matrix dimension ', matrix.shape)


# kmeans

kmeans = KMeans(n_clusters=200, random_state=1).fit(matrix)


# get all products per per cluster, add cluster values for each row in joined table

X_training_reduced_data_matrix['cluster_id'] = pandas.DataFrame(kmeans.labels_)
training_user_join = pandas.merge(training_user_join, X_training_reduced_data_matrix[['user_id', 'cluster_id']])
del X_training_reduced_data_matrix
print('\n\n\n\ntraining_user_join\n\n', training_user_join.head(20).to_string()
      , '\n\ntraining_user_join dimension ', training_user_join.shape)


# create a list of products in each cluster

products_per_cluster_dataframe = training_user_join.groupby(by=['cluster_id'], as_index=False).agg({'product_id': lambda x: list(x)})

print('\n\n\n\nproducts_per_cluster_dataframe\n\n', products_per_cluster_dataframe.head(20).to_string()
      , '\n\nproducts_per_cluster_dataframe dimension ', products_per_cluster_dataframe.shape)


# get top 10 products from each cluster

products_per_cluster_dataframe['recommendation_product_id'] = products_per_cluster_dataframe['product_id'].apply(lambda x: Counter(w for w in x).most_common(10))
products_per_cluster_dataframe['recommendation_product_id'] = products_per_cluster_dataframe['recommendation_product_id'].apply(lambda x: [ i[0] for i in x])
products_per_cluster_dataframe = products_per_cluster_dataframe.drop(['product_id'], axis=1)
print('\n\n\n\ncluster_product_aggregate top 10 products for each cluster\n\n', products_per_cluster_dataframe.head(20).to_string(),
      '\n\n')


# kmeans prediction on validation feature matrix


h = pandas.DataFrame()
h['user_id'] = X_validation_reduced_data_matrix['user_id']
validation_matrix = X_validation_reduced_data_matrix.drop(['user_id'], axis=1)


validation_clusters = kmeans.predict(validation_matrix)


X_validation_reduced_data_matrix['user_id'] = h['user_id']
X_validation_reduced_data_matrix['cluster_id'] = pandas.DataFrame(validation_clusters)

print('\n\n\n\n validation cluster array length ', len(validation_clusters))
apple=0
for i in validation_clusters:
    if i is None:
        apple+=1
print('apple ',apple)
validation_comparison_matrix = pandas.merge(X_validation_reduced_data_matrix[['user_id', 'cluster_id']], products_per_cluster_dataframe, on='cluster_id')

print('\n\n\n\na\n\n', validation_comparison_matrix.head(20).to_string(), '\n\na dimension ', validation_comparison_matrix.shape)
print('\n\nX_validation_reduced_data_matrix dimension ', X_validation_reduced_data_matrix.shape)
print('\n\nproducts_per_cluster_dataframe dimension ', products_per_cluster_dataframe.shape)

print('\n\n\n\nX_validation_reduced_data_matrix\n\n', X_validation_reduced_data_matrix.head(20).to_string())
b = pandas.merge(X_validation_reduced_data_matrix, pandas.merge(orders,training_order_products) )
b = b.groupby(['user_id'], as_index=False)
b = b.agg({'product_id': lambda  x: list(x)})

print('\n\n\n\nb\n\n', b.head(20).to_string(), '\n\nb dimension ', b.shape)

validation_comparison_matrix = pandas.merge(validation_comparison_matrix, b)

print('\n\n\n\nvalidation_comparison_matrix\n\n', validation_comparison_matrix.head(20).to_string(), '\n\nvalidation_comparison_matrix dimension ', validation_comparison_matrix.shape)


list1 = list()
for i in validation_comparison_matrix['product_id'].iteritems():
    list1.append(i)
list2 = list()
for i in validation_comparison_matrix['recommendation_product_id'].iteritems():
    list2.append(i)

list3 = zip(list1, list2)

list4 = list()
for i in list3:
    ce=0
    s0 = set()
    for item in i[0][1]:
        s0.add(item)
    s1 = set()
    for item in i[1][1]:
        s1.add(item)
    for x in s0:
        if s1.__contains__(x):
            ce+=1
    list4.append(ce)

print(len(list1))
print(len(list2))
print('\n\nabc ', sum(list4) / ( len(list4) * 10 ) )

print('\n\n\n\na\n\n', validation_comparison_matrix.head(20).to_string())

c = validation_comparison_matrix.groupby(by=True)

c['compare_list'] = c['compare_list'].mean()

print('\n\n\n\nc\n\n', c.head(20).to_string(), '\n\ndimension ', c.shape)
























#
#
#
# # test users
#
# test_user_grouped = test_user_join.groupby(['user_id'], as_index=False)
# test_user_dataframe = test_user_grouped.agg({'department_id': lambda x: list(x),
#                               'order_hour_of_day': lambda x: list(x),
#                               'days_since_prior_order': lambda x: list(x),
#                               'order_dow': lambda x: list(x),
#                               'reordered': lambda x: list(x)
#                                                })
#
# reduced_test_data_matrix = pandas.DataFrame(data=None, columns=['user_id'])
# reduced_test_data_matrix['user_id'] = test_user_dataframe['user_id']
# for i in range(1, 22):
#     reduced_test_data_matrix['department_c_' + str(i)] = test_user_dataframe['department_id'].apply(lambda x: x.count(i))
# for i in [1,2,3]:
#     reduced_test_data_matrix['hour_of_day_c_' + str(i)] = test_user_dataframe['order_hour_of_day'].apply(lambda x: x.count(i))
# for i in [1,2]:
#     reduced_test_data_matrix['day_of_week_c_' + str(i)] = test_user_dataframe['order_dow'].apply(lambda x: x.count(i))
# for i in [1, 2, 3, 4]:
#     reduced_test_data_matrix['prior_order_days_c_' + str(i)] = test_user_dataframe['days_since_prior_order'].apply(lambda x: x.count(i))
# for i in [1,2]:
#     reduced_test_data_matrix['order_reorder_c_' + str(i)] = test_user_dataframe['reordered'].apply(lambda x: x.count(1))
# del test_user_dataframe
#
# # test feature matrix - each row contain feature values for 1 user      total length 75000      Number of columns (features) 32
# test_matrix = reduced_test_data_matrix.drop(['user_id'], axis=1)
# print('\n\n\n\ntest feature matrix\n\n', test_matrix.head(20).to_string())
# print('\n\n\n\ntest feature matrix dimension:', test_matrix.shape)
#
#
#
#
# # Kmean predict()
# # predict cluster for each test user
#
# kmeans.predict(test_matrix)
#
# test_users_predicted_clusters = kmeans.labels_
# print('\n\n\n\n kmeans labels length\n', len(kmeans.labels_))
# print('\n\n\n\ntest_users_predicted_clusters\n\n', test_users_predicted_clusters)
#
# reduced_test_data_matrix['cluster_id'] = pandas.DataFrame(kmeans.labels_)
# a = pandas.merge(products_per_cluster_dataframe, reduced_test_data_matrix, on='cluster_id')
# orders = pandas.read_csv(instacart_data_location + 'orders.csv', usecols=['order_id', 'user_id', 'eval_set'])
# a = pandas.merge(a, orders)
# a = a[a['eval_set']=='test']
# a = a[['order_id', 'product_id']]
#
# # cluster the output of the training set into n (current value = 50) cluster to to get n new classes to classify the training users and test users, then use svc to classify the test users.
#
# training_product_orders_join = pandas.merge(orders, training_order_products)
#
# print('\ntraining_product_orders_join shape ', training_product_orders_join.shape)
# products = pandas.read_csv(instacart_data_location + 'products.csv', usecols=['product_id', 'aisle_id', 'department_id'])
# training_product_orders_join = pandas.merge(training_product_orders_join, products)
#
# del products, training_order_products
#
# training_product_orders_grouped = training_product_orders_join.groupby(['user_id'], as_index=False)
#
# user_aggregate1 = training_product_orders_grouped.agg({'aisle_id': lambda x: list(x),
#                                                        'department_id': lambda x: list(x)})
# del training_product_orders_grouped
# print('\n\n\n\nuser aggregate dimensions ', user_aggregate1.shape)
#
# # training output product data matrix to cluster the training output in n(current value=50) cluster(output classes)
# training_set_output_df_matrix = pandas.DataFrame()
# training_set_output_df_matrix['user_id'] = user_aggregate1['user_id']
#
# for i in range(1, 135):
#     training_set_output_df_matrix['aisle_c_' + str(i)] = user_aggregate1['aisle_id'].apply(lambda x: x.count(i))
# # for i in range(1, 22):
# #     training_set_output_df_matrix['department_c_' + str(i)] = user_aggregate1['department_id'].apply(lambda x: x.count(i))
#
# class_matrix = training_set_output_df_matrix.drop(['user_id'], axis=1)
# print('\n\n\n\nclass_matrix\n\n', class_matrix.head(20).to_string())
# print('\n\nclass_matrix dimensions', class_matrix.shape)
#
#
#
#
# # kmeans on training output product data matrix
# training_output_kmeans = KMeans(n_clusters=200, random_state=1).fit(class_matrix)
#
#
#
#
# # user_classification_list = zip(training_set_output_df_matrix.user_id, training_output_kmeans.labels_)
# # user_classification_list.to_csv(location + 'user_classification_list.csv', index=False)
#
# training_set_output_df_matrix['cluster_id'] = pandas.DataFrame(training_output_kmeans.labels_)
# training_set_output_df_matrix = training_set_output_df_matrix.sort_values(['user_id'])
#
# training_product_orders_join = pandas.merge(training_product_orders_join, training_set_output_df_matrix[['user_id', 'cluster_id']])
# training_product_orders_grouped = training_product_orders_join.groupby(['cluster_id'], as_index=False)
#
# training_product_orders_aggregate = training_product_orders_grouped.agg({'product_id': lambda x: list(pandas.Series(x).value_counts())})
# training_product_orders_join = pandas.merge(training_product_orders_join, training_set_output_df_matrix[['user_id', 'cluster_id']])
# print(6)
