import pandas as pd
import numpy as np
import os
from collections import defaultdict
import scipy.sparse as sparse

def read_poi_coos(poi_file):
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos

def read_training_data(train_file, user_num, poi_num, config):
    train_data = open(train_file, 'r').readlines()
    if config['model'] == 'UserBasedCF' or config['model'] == 'ItemBasedCF':
        training_matrix = np.zeros((user_num, poi_num))
    elif config['model'] == 'MF':
        training_matrix = sparse.dok_matrix((user_num, poi_num))
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        training_matrix[uid, lid] = 1.0
    return training_matrix

def read_ground_truth(test_file):
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth

def mf_read_train_test_datasets(num_users, num_items, train_path, test_path, user_num2id):
    sparse_training_matrix = sparse.dok_matrix((num_users, num_items))
    ground_truth = defaultdict(list)
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    for row in train_dataset.iterrows():
        row = row[1]
        u_id = int(user_num2id[row['user_pin']])
        p_id = int(row['shop_id'])
        sparse_training_matrix[u_id, p_id] = 1.0

    for row in test_dataset.iterrows():
        row = row[1]
        u_id = int(user_num2id[row['user_pin']])
        p_id = int(row['shop_id'])
        ground_truth[u_id].append(p_id)

    return sparse_training_matrix, ground_truth

def cf_read_train_test_datasets(num_users, num_items, train_path, test_path, user_num2id):
    user_item_matrix = np.zeros((num_users, num_items))
    ground_truth = defaultdict(list)
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    for row in train_dataset.iterrows():
        row = row[1]
        u_id = int(user_num2id[row['user_pin']])
        p_id = int(row['shop_id'])
        user_item_matrix[u_id, p_id] = 1.0

    for row in test_dataset.iterrows():
        row = row[1]
        u_id = int(user_num2id[row['user_pin']])
        p_id = int(row['shop_id'])
        ground_truth[u_id].append(p_id)

    return user_item_matrix, ground_truth

def origin_id_trans(origin_set):
    ## 为origin_set中的值分配编号
    origin2id = dict()
    id2origin = dict()
    for index, origin in enumerate(origin_set):
        origin2id[origin] = index
        id2origin[index] = origin

    return origin2id, id2origin

def data_load(user_path, shop_path, order_path, order_all_online_path, order_all_offline_path):

    ####  从数据库中加载数据  ####
    if not os.path.exists(order_all_offline_path):
        read_sql(user_path, 'select * from dmu_dev.cross_rec_wfj_uer')
        read_sql(shop_path, 'select shop_num, shop_name, map_lng, map_lat, address, one_industry, two_industry from dmu_dev.cross_rec_wfj_shop')
        read_sql(order_path, 'select pin, shop_num, shop_name, ord_amount, order_id1, order_id2, pay_finish_time from dmu_dev.cross_rec_wfj_order')
        read_sql(order_all_online_path, 'select user_pin, sale_ord_det_id, sale_ord_tm, actual_pay_amount, receive_city_name, item_id, shop_id, item_first_cate_cd, item_first_cate_name, item_second_cate_cd, item_second_cate_name from dmu_dev.cross_rec_wfj_one_year_online_order')
        read_sql(order_all_offline_path, 'select pin, shop_no, one_industry, two_industry, ord_amount, order_id1, order_id2, pay_finish_time from dmu_dev.cross_rec_wfj_one_year_offline_order')
        print('data load success')

def read_sql(data_path, sql): 
    
    ##  用sql语句 读取数据 并存放到data_path中
    input_file_path = data_path
    read_data_sql = """hive -e "set hive.cli.print.header=true;
    {s} " > {path}
    """.format(s=sql, path=input_file_path)
    res_sql = os.system(read_data_sql)
    if res_sql == 0:
        print("Read data done!")
    else:
        raise ValueError("输入数据读取失败!")
        
    print("write {} data done!".format(data_path))
    
def data_drop(user_path, shop_path, order_path, order_all_online_path, order_all_offline_path):
    
    #### 去重去空 ####
    
    ## user
    user = pd.read_csv(user_path, sep='\t')
    user = user.drop_duplicates(['user_pin']).reset_index(drop=True)
    user = user.dropna(subset=['user_pin'], axis=0).reset_index(drop=True)
    print('user 去重去空', user)
    
    ##  shop
    shop = pd.read_csv(shop_path, sep='\t')
    shop = shop.drop_duplicates(['shop_num']).reset_index(drop=True)
    shop = shop.dropna(subset=['shop_num'], axis=0).reset_index(drop=True)
    print('shop 去重去空', shop)

    ##  order
    order = pd.read_csv(order_path, sep = '\t')
    order = order.drop_duplicates(subset=['order_id1', 'order_id2'])
    order = order.dropna(subset=['pin'], axis=0)
    print('order 去重去空', order)
    
    ##  order all online
    order_all_online = pd.read_csv(order_all_online_path, sep = '\t')
    order_all_online = order_all_online.drop_duplicates(subset=['sale_ord_det_id'])
    order_all_online = order_all_online.dropna(subset=['user_pin'], axis=0)
    print('order_all_online 去重去空', order_all_online)

    ##  order all offline
    order_all_offline = pd.read_csv(order_all_offline_path, sep = '\t')
    order_all_offline = order_all_offline.drop_duplicates(subset=['order_id1', 'order_id2'])
    order_all_offline = order_all_offline.dropna(subset=['pin'], axis=0)
    print('order_all_offline 去重去空', order_all_offline)
    
    return user, shop, order, order_all_online, order_all_offline

def data_filter(user, shop, order, order_all_online, order_all_offline):
    
    #### 筛选有用的数据，并统一字段名 ####
    ## order
    # 统一字段名
    order = order.rename(columns = {"pin": "user_pin"})
    #  哆啦宝（聚合支付）支持微信支付、支付宝、京东支付等多种付款方式，订单中涉及的用户虽然都有京东PIN，但其中只有部分原京东用户有画像信息，大部分京东PIN是新创建的，没有画像信息。
    #  筛选出有画像的用户对应的订单
    order = order[order['user_pin'].isin(user['user_pin']) & order['shop_num'].isin(shop['shop_num'])].reset_index(drop=True)

    # 按时间排序
    order = order.sort_values(by = 'pay_finish_time')
    order = order.reset_index(drop=True)
    print('order filter', order)
    
    ## shop
    # 筛选出有交易记录（正在经营）的店铺
    shop = shop[shop['shop_name'].isin(order.shop_name)]
    shop = shop.reset_index(drop=True)
    print('shop filter', shop)
    
    ## user
    user = user.drop(columns = ['dt', 'user_num'])
    user = user[user['user_pin'].isin(order.user_pin)]
    user = user.reset_index(drop=True)
    print('user filter', user)
    
    ## order all offline
    # 筛选有画像用户对应的线上订单
    order_all_online = order_all_online[order_all_online['user_pin'].isin(user['user_pin'])].reset_index(drop=True)
    print('order_all_online filter', order_all_online)

    ## order all offline
    # 筛选有画像用户对应的线下订单
    order_all_offline = order_all_offline.rename(columns = {"pin": "user_pin"})
    order_all_offline = order_all_offline[order_all_offline['user_pin'].isin(user['user_pin'])].reset_index(drop=True)
    print('order_all_offline filter', order_all_offline)
    
    return user, shop, order, order_all_online, order_all_offline

if __name__ == '__main__':
    
    read_folder = 'data/'
    user_path = read_folder + 'sql_data/user_info'
    shop_path = read_folder + 'sql_data/shop_info'
    order_path = read_folder + 'sql_data/orders_info'
    order_all_online_path = read_folder + 'sql_data/order_online_info'
    order_all_offline_path = read_folder + 'sql_data/order_offline_info'
    
    #### 从数据库中加载数据
    data_load(user_path, shop_path, order_path, order_all_online_path, order_all_offline_path)
    #### 去重去空
    user, shop, order, order_all_online, order_all_offline = data_drop(user_path, shop_path, order_path, order_all_online_path, order_all_offline_path)
    #### 筛选有用的数据，并统一字段名
    user, shop, order, order_all_online, order_all_offline = data_filter(user, shop, order, order_all_online, order_all_offline)
    
    
    
    