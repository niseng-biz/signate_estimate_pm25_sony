#-------------------------------------------------------------------------------
# Author:   niseng
# Created:  14/05/2022
# lisence:  CC0
#-------------------------------------------------------------------------------

import folium
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import pickle
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from dtreeviz.trees import *

class config:
    file_rev = 'rev90'
    random_seed = 2022
    nfold = 6
    target = 'pm25_mid'
    kfoldfeature = 'timestamp_label_enc'
    drop_soukan = 1.0
    foptuna = 0
    cluster_num = 40
    shrink_factor = 1.05

    #毎回やらなくてもいいもの
    fclustering = 1
    fmakedailyslice = 1
    fmakemap = 1

    #毎回やるもの
    fPreprocess = 0
    fTraining = 1
    fVisualize = 1
    fPrediction = 1

    #明らかに遠い観測点を除去
    DropCitylist = ['Novosibirsk','Darwin', 'Perth','Ürümqi', 'Naha', 'Calama', 'Sapporo', 'Hegang', 'Bandar Abbas', 'Yazd']


if config.foptuna == 0:
    import lightgbm as lgb
    lgbm_numleaves = 70
    num_boost_rounds_set = 10000
else:
    import optuna.integration.lightgbm as lgb
    lgbm_numleaves = 31
    num_boost_rounds_set = 200

model_path = "model/"
data_path = "data/"
modelfilename = 'model_'+ config.file_rev +'.pkl'
oofvalidname = 'oofvalid_'+ config.file_rev +'.pkl'
submitfilename = 'submit_'+ config.file_rev +'.csv'
featureimortancename = 'fearture_importance'


def read_data():

    cluster_data = pd.read_csv(data_path + 'cluster_cities' + str(config.cluster_num) + '_' +str(config.shrink_factor) + '.csv')
    train_data = pd.read_csv(data_path + 'train_add.csv')
    test_data = pd.read_csv(data_path + 'test_add.csv')

    return train_data, test_data, cluster_data


def make_map():
    # データの読み込み(各々読み込んでください)
    train_data = pd.read_csv(data_path + 'train.csv', index_col='id')
    test_data = pd.read_csv(data_path + 'test.csv', index_col='id')

    # train_dataとtest_dataの緯度・経度・都市名(ユニーク値)を取ってきて結合
    cities_train = train_data[['lat','lon','City']].drop_duplicates()
    cities_test = test_data[['lat','lon','City']].drop_duplicates()
    cities_total = pd.concat([cities_train,cities_test]).reset_index(drop=True)

    # 皇居スタート
    map = folium.Map(location=[35.6852,139.7528], zoom_start=8)

    # train_dataのピンは青、test_dataのピンは赤
    for i, r in cities_total.iterrows():
      if i < len(cities_train):
        folium.Marker(location=[r['lat'], r['lon']], popup=r['City']).add_to(map)
      else:
        folium.Marker(location=[r['lat'], r['lon']], popup=r['City'],icon=folium.Icon(color='red')).add_to(map)

    # draw line
    # longitude direction
    line_div = 10
    line_num = int(180/line_div)
    for i in range(line_num):
        points = []
        points.append(tuple([90,  line_div * i + line_div/2]))
        points.append(tuple([-90, line_div * i + line_div/2]))
        folium.PolyLine(points, color="blue", weight=1.5, opacity=1).add_to(map)
        points = []
        points.append(tuple([90,  -(line_div * i + line_div/2)]))
        points.append(tuple([-90, -(line_div * i + line_div/2)]))
        folium.PolyLine(points, color="blue", weight=1.5, opacity=1).add_to(map)

    # latitude direction
    line_div = 10
    line_num = int(90/line_div)
    for i in range(line_num):
        points = []
        points.append(tuple([line_div * i,-180]))
        points.append(tuple([line_div * i, 180]))
        folium.PolyLine(points, color="blue", weight=1.5, opacity=1).add_to(map)
        points = []
        points.append(tuple([-line_div * i,-180]))
        points.append(tuple([-line_div * i, 180]))
        folium.PolyLine(points, color="blue", weight=1.5, opacity=1).add_to(map)

    map.save("map_train_test_div10_lat offset.html")


def position_clustering():
    # データの読み込み(各々読み込んでください)
    train_data = pd.read_csv(data_path + 'train.csv', index_col='id')
    test_data = pd.read_csv(data_path + 'test.csv', index_col='id')
    cities_train = train_data[['lat','lon','City']].drop_duplicates()
    cities_test = test_data[['lat','lon','City']].drop_duplicates()
    cities_total = pd.concat([cities_train,cities_test]).reset_index(drop=True)

    print(cities_total)

    for x in config.DropCitylist: cities_total = cities_total[~(cities_total["City"]== x)]

    cities_total.set_index('City', inplace = True)

    print(cities_total)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100, ncols=2)
    ax[0].scatter(cities_total["lon"], cities_total["lat"],marker="o", facecolor="none",
               edgecolors="black", s=80)
    ax[0].set_title('before clustering')
    ax[0].set_xlabel('longitude')
    ax[0].set_ylabel('latitude')

    # latに係数をかけて計算
    cities_total['lat'] = cities_total['lat'] / config.shrink_factor

    kmeans = KMeans( n_clusters = config.cluster_num, random_state = 0 )
    result = kmeans.fit(cities_total)
    output = kmeans.predict(cities_total)

    # latを元に戻す
    cities_total['lat'] = cities_total['lat'] * config.shrink_factor

    ax[1].scatter(cities_total["lon"], cities_total["lat"], c=result.labels_)
    ax[1].scatter(result.cluster_centers_[:,1],   # km.cluster_centers_には各クラスターのセントロイドの座標が入っている
                    result.cluster_centers_[:,0]*config.shrink_factor,
                    s=100,
                    marker='x',
                    c='red')
    ax[1].set_title('after clustering')
    ax[1].set_xlabel('longitude')
    ax[1].set_ylabel('latitude')

    cities_total = cities_total.assign(clu_num = [x  for x in result.labels_])

    for i in range(len(result.cluster_centers_[:,0])):
        cities_total.loc[cities_total['clu_num'] == i, 'clu_lat'] = result.cluster_centers_[i,0] * config.shrink_factor
        cities_total.loc[cities_total['clu_num'] == i, 'clu_lon'] = result.cluster_centers_[i,1]

    cities_total.to_csv(data_path + 'cluster_cities' + str(config.cluster_num) + '_' +str(config.shrink_factor) + '.csv')
    print(cities_total)

    plt.show()

    return


def make_sub_df(_df, lon_this_city, lat_this_city, lat_bound_init, lon_bound_init ):
    # lon, latで切り取り
    lon_bound = lon_bound_init
    lat_bound = lat_bound_init

    counter = 0
    _df2 = pd.DataFrame()
    while((len(_df2) < 5) & (counter < 10)):
        _df2 = _df[_df['lon']  > (lon_this_city[0] - lon_bound)]
        _df2 = _df2[_df2['lon']  < (lon_this_city[0] + lon_bound)]
        _df2 = _df2[_df2['lat']  > (lat_this_city[0] - lat_bound)]
        _df2 = _df2[_df2['lat']  < (lat_this_city[0] + lat_bound)]
        lon_bound = lon_bound + 10
        lat_bound = lat_bound + 10
        counter = counter + 1

    _df2 = _df2.reset_index()

    return _df2

def sort_df(_df2, city_name, base_co_mid, base_no2_mid, co_weight):

    if (len(base_co_mid) == 0) & (len(base_no2_mid) == 0):
        for i in range(len(_df2)):
            original = (_df2.loc[_df2['City'] == city_name, 'lat'].values, _df2.loc[_df2['City'] == city_name, 'lon'].values)
            destination = (_df2.loc[i, ['lat']].values, _df2.loc[i, ['lon']].values)
            x = geodesic(original, destination).km
            _df2.at[0+i, 'indicator'] = x

        _df2 = _df2.sort_values('indicator', ascending=True).reset_index(drop=True)
        print(city_name)
        print('There are no co_mid and no2_mid infomation. Use lon, lat')
        print('Distance is ' + str(_df2.loc[1,['indicator']].values))
    else:
        _df2['co_mid_dif'] = abs(_df2['co_mid'] - base_co_mid[0])
        _df2['no2_mid_dif'] = abs(_df2['no2_mid'] - base_no2_mid[0])
        _df2['indicator'] = _df2['co_mid_dif'] * co_weight + _df2['no2_mid_dif'] * (1-co_weight)
        _df2 = _df2.sort_values('indicator', ascending=True).reset_index(drop=True)

    return _df2

def make_dailyslice():
    print('making daily slices ...')

    train_data = pd.read_csv(data_path + 'train.csv')
    test_data = pd.read_csv(data_path + 'test.csv')
    train_data = train_data.assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))
    test_data = test_data.assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))

    # lon, latで切り取り
    lon_bound_init = 10
    lat_bound_init = 10
    co_weight = 0.8
    run_counter = 0
    err_counter = 0

    for timestamp_val in train_data['timestamp'].unique():

        # ファイルの存在確認
        if os.path.isfile(data_path + 'train_dailyslice_save_'+pd.to_datetime(timestamp_val).strftime('%Y-%m-%d') +'.csv') & os.path.isfile(data_path + 'test_dailyslice_save_'+pd.to_datetime(timestamp_val).strftime('%Y-%m-%d') +'.csv') :
            pass
        else:
            # timestampで切り取り
            print(timestamp_val)
            timestamp_val = pd.to_datetime(timestamp_val)
            _df_train = train_data.loc[train_data['timestamp'] == timestamp_val, :]
            _df_test = test_data.loc[test_data['timestamp'] == timestamp_val, :]

            for city_name in _df_train['City']:
                run_counter = run_counter + 1
                lon_this_city = _df_train.loc[_df_train['City'] == city_name, 'lon'].values
                lat_this_city = _df_train.loc[_df_train['City'] == city_name, 'lat'].values

                # 周辺領域のデータをきりだし
                _df2 = make_sub_df(_df_train, lon_this_city, lat_this_city, lat_bound_init, lon_bound_init)

                base_co_mid = _df_train.loc[_df_train['City'] == city_name, 'co_mid'].values
                base_no2_mid = _df_train.loc[_df_train['City'] == city_name, 'no2_mid'].values

                _df2 = sort_df(_df2, city_name, base_co_mid, base_no2_mid, co_weight)

                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'co_mid_neighbor'] = _df2['co_mid'][1]
                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'no2_mid_neighbor'] = _df2['no2_mid'][1]
                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'pm25_mid_neighbor'] = _df2['pm25_mid'][1]
                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'City_neighbor'] = _df2['City'][1]
                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'lat_dif_neighbor'] = _df2['lat'][1] - lat_this_city[0]
                train_data.loc[(train_data['City'] == city_name) & (train_data['timestamp'] == timestamp_val), 'lon_dif_neighbor'] = _df2['lon'][1] - lon_this_city[0]

            for city_name in _df_test['City']:
                run_counter = run_counter + 1
                lon_this_city = _df_test.loc[_df_test['City'] == city_name, 'lon'].values
                lat_this_city = _df_test.loc[_df_test['City'] == city_name, 'lat'].values

                # 周辺領域のデータをきりだし
                _df2 = make_sub_df(_df_train, lon_this_city, lat_this_city, lat_bound_init, lon_bound_init)

                base_co_mid = _df_test.loc[_df_test['City'] == city_name, 'co_mid'].values
                base_no2_mid = _df_test.loc[_df_test['City'] == city_name, 'no2_mid'].values

                _df2 = sort_df(_df2, city_name, base_co_mid, base_no2_mid, co_weight)

                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'co_mid_neighbor'] = _df2['co_mid'][1]
                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'no2_mid_neighbor'] = _df2['no2_mid'][1]
                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'pm25_mid_neighbor'] = _df2['pm25_mid'][1]
                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'City_neighbor'] = _df2['City'][1]
                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'lat_dif_neighbor'] = _df2['lat'][1] - lat_this_city[0]
                test_data.loc[(test_data['City'] == city_name) & (test_data['timestamp'] == timestamp_val), 'lon_dif_neighbor'] = _df2['lon'][1] - lon_this_city[0]


            train_data.loc[train_data['timestamp'] == timestamp_val, :].to_csv(data_path + 'train_dailyslice_save_'+pd.to_datetime(timestamp_val).strftime('%Y-%m-%d') +'.csv')
            test_data.loc[test_data['timestamp'] == timestamp_val, :].to_csv(data_path + 'test_dailyslice_save_'+pd.to_datetime(timestamp_val).strftime('%Y-%m-%d') +'.csv')

    train_data.to_csv(data_path + 'train_add.csv')
    test_data.to_csv(data_path + 'test_add.csv')

    print('finish making daily slices')

    return



def Preprocessing( train_data, test_data, cluster_data):
    #前処理
    #説明変数と目的関数の確認
    train_labels=list(train_data.columns)
    test_labels=list(test_data.columns)

    for x in config.DropCitylist: train_data = train_data[~(train_data["City"]== x)]
    train_data = train_data.reset_index()

    train_data = train_data.assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))
    train_data = train_data.assign(weeknum=[str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1])  for x in train_data['timestamp'].tolist()])

    test_data = test_data.assign(timestamp=lambda d: pd.to_datetime(d['year'].astype(str) + '-' + d['month'].astype(str) + '-' + d['day'].astype(str)))
    test_data = test_data.assign(weeknum=[str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1])  for x in test_data['timestamp'].tolist()])

    #緯度・経度が少しずれていて気持ち悪いので１つの値にする
    cities_train = train_data[['lat','lon','City']].drop_duplicates().reset_index(drop=True)
    cities_test = test_data[['lat','lon','City']].drop_duplicates().reset_index(drop=True)

    train_data['clu_num'] = ""
    train_data['clu_lat'] = ""
    train_data['clu_lon'] = ""
    test_data['clu_num'] = ""
    test_data['clu_lat'] = ""
    test_data['clu_lon'] = ""

    for city_name in cities_train['City']:
        clu_num = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_num']].values)
        clu_lat = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_lat']].values)
        clu_lon = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_lon']].values)
        city_lat = float(cities_train.loc[(cities_train['City'] == city_name),['lat']].values)
        city_lon = float(cities_train.loc[(cities_train['City'] == city_name),['lon']].values)

        train_data.loc[train_data['City'] == city_name,['clu_num']] = clu_num
        train_data.loc[train_data['City'] == city_name,['clu_lat']] = clu_lat
        train_data.loc[train_data['City'] == city_name,['clu_lon']] = clu_lon
        train_data.loc[train_data['City'] == city_name,['lat']] = city_lat
        train_data.loc[train_data['City'] == city_name, ['lon']] = city_lon

    for city_name in cities_test['City']:
        clu_num = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_num']].values)
        clu_lat = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_lat']].values)
        clu_lon = float(cluster_data.loc[cluster_data['City'] == city_name, ['clu_lon']].values)
        city_lat = float( cities_test.loc[(cities_test['City'] == city_name),['lat']].values)
        city_lon = float( cities_test.loc[(cities_test['City'] == city_name),['lon']].values)

        test_data.loc[test_data['City'] == city_name,['clu_num']] = clu_num
        test_data.loc[test_data['City'] == city_name,['clu_lat']] = clu_lat
        test_data.loc[test_data['City'] == city_name,['clu_lon']] = clu_lon
        test_data.loc[test_data['City'] == city_name, 'lat'] = city_lat
        test_data.loc[test_data['City'] == city_name, 'lon'] = city_lon


    train_data.to_csv('train_data_for_check.csv')
    print(train_data.head())
    test_data.to_csv('test_data_for_check.csv')
    print(test_data.head())

    # type conversion is needed. inserted value type was object.
    train_data = train_data.astype({'clu_num': 'int64', 'clu_lat': 'float64', 'clu_lon': 'float64', 'lon': 'float64', 'lat': 'float64'})
    test_data = test_data.astype({'clu_num': 'int64', 'clu_lat': 'float64', 'clu_lon': 'float64', 'lon': 'float64', 'lat': 'float64'})

    train_data['timestamp'] = train_data['timestamp'].dt.strftime('%Y-%m-%d')
    test_data['timestamp'] = test_data['timestamp'].dt.strftime('%Y-%m-%d')

    # 単純に相関を見てみる
    corr_pm25_mid = train_data.corr(method='pearson')[config.target].abs()
    print(corr_pm25_mid.sort_values(ascending=False)[1:16])
    high_corr = corr_pm25_mid.sort_values(ascending=False)[:35].index
    print(high_corr)
    df_ = train_data[high_corr]
    dispname = [x.split(":")[0] for x in df_.columns.values]

    # corr_top10はfeaturemapにのみ使用。
    corr_top10 = high_corr[1:11].tolist()
    print(corr_top10)

    ### target encording ###
    train_data = train_data.assign(clu_num_month=lambda d: (d['clu_num'].astype(str) + '-' + d['year'].astype(str) + '-' + d['month'].astype(str)  ))
    test_data = test_data.assign(clu_num_month=lambda d: (d['clu_num'].astype(str) + '-' + d['year'].astype(str) + '-' + d['month'].astype(str) ))

    kf = KFold(n_splits = 4, shuffle=True, random_state=config.random_seed)
    tmp_train = pd.Series(np.empty(train_data.shape[0]), index=train_data.index)
    tmp_test = pd.Series(np.empty(test_data.shape[0]), index=test_data.index)


    counter = 0

    for c in train_data['clu_num_month'].unique():
        # all target mean for test
        train_index = (train_data['clu_num_month'] == c)
        test_index = (test_data['clu_num_month'] == c)


        mean_val_tmp = train_data.loc[train_data['clu_num_month'] == c,[config.target]].values.mean()
        test_data.loc[test_data['clu_num_month'] == c,'clu_num_month'] = mean_val_tmp
        sub_df = train_data.loc[train_data['clu_num_month'] == c,:]

        for idx_1, idx_2 in kf.split(sub_df):
            id_test = sub_df.iloc[idx_1, sub_df.columns.get_loc('id')]
            id_valid = sub_df.iloc[idx_2, sub_df.columns.get_loc('id')]

            mean_val_tmp = 0
            mean_val_tmp = [train_data.loc[train_data['id'] == x,[config.target]].values for x in id_test]
            mean_val_tmp = sum(mean_val_tmp)
            mean_val_tmp = mean_val_tmp / len(id_test)

            for x in id_valid: train_data.loc[train_data['id'] == x,['clu_num_month']] = mean_val_tmp

        counter = counter +1
        print('loop count',counter)

    train_data = train_data.astype({'clu_num_month': 'float64'})
    test_data = test_data.astype({'clu_num_month': 'float64'})

    print(train_data)
    print(test_data)

    #一旦selectedfeatureやめてみる
    SelectedTrainData = train_data
    SelectedTestData = test_data


    numeric_columns = [c for c in SelectedTrainData.columns if re.search(r'max|mid|min|cnt|var|lat|lon|clu|month|co_mid_neighbor|no2_mid_neighbor|pm25_mid_neighbor|lat_dif_neighbor|lon_dif_neighbor', c) and not re.fullmatch(r'pm25_mid', c)]
    categorical_columns = ['weeknum','timestamp']
    agg_country_value = dict([(c, [np.mean, np.max, np.min, np.std]) for c in corr_top10])

    run_blocks = [
        *[NumericFeatBlock(c) for c in [numeric_columns]],
        *[CategoricalFeatBlock(c) for c in categorical_columns],
        #*[DateFeatureBlock()],
        *[AggregateValueBlock(['clu_num', 'weeknum'], agg_country_value)],
    ]

    train_y = SelectedTrainData[config.target]
    train_X = get_train_data(SelectedTrainData, run_blocks, fit_df=pd.concat([SelectedTrainData, SelectedTestData], ignore_index=True))
    test_X = get_train_data(SelectedTestData, run_blocks, fit_df=pd.concat([SelectedTrainData, SelectedTestData], ignore_index=True))

    train_X.to_csv(data_path + 'train_X_' + config.file_rev + '.csv')
    train_y.to_csv(data_path + 'train_y_' + config.file_rev + '.csv')
    test_X.to_csv(data_path + 'test_X_' + config.file_rev + '.csv')


    if config.foptuna == 1:
        drop_columns_list = ['agg_clu_num_weeknum_so2_max_amin', 'agg_clu_num_weeknum_so2_mid_amin', 'agg_clu_num_weeknum_o3_max_std', 'agg_clu_num_weeknum_ws_mid_amax', 'agg_clu_num_weeknum_no2_mid_amax',
                            'agg_clu_num_weeknum_o3_max_amax', 'agg_clu_num_weeknum_no2_min_amin', 'agg_clu_num_weeknum_so2_mid_amax', 'agg_clu_num_weeknum_no2_max_amax', 'agg_clu_num_weeknum_co_max_amin',
                            'agg_clu_num_weeknum_no2_min_amax', 'agg_clu_num_weeknum_co_max_std', 'agg_clu_num_weeknum_no2_min_std', 'agg_clu_num_weeknum_co_mid_amin', 'agg_clu_num_weeknum_so2_max_amax',
                            'agg_clu_num_weeknum_so2_max_std', 'agg_clu_num_weeknum_co_max_amax', 'agg_clu_num_weeknum_co_min_amin', 'timestamp_label_enc']

        train_X = train_X.drop(columns = drop_columns_list)
        test_X = test_X.drop(columns = drop_columns_list)


    print(train_X.columns.tolist())
    print(train_y.name)
    print(test_X.columns.tolist())

    return train_X, train_y, test_X

def read_preprocessed_data():

    train_X = pd.read_csv(data_path + 'train_X_' + config.file_rev + '.csv')
    train_y = pd.read_csv(data_path + 'train_y_' + config.file_rev + '.csv')
    test_X  = pd.read_csv(data_path + 'test_X_' + config.file_rev + '.csv')

    return train_X, train_y, test_X


def group_kfold_split(df, col_group, n_splits, random_state):

    group_id = df[col_group].value_counts().reset_index()
    group_id.columns = [col_group, 'count']
    group_id_list = group_id[col_group].unique()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf_list = list(kf.split(group_id_list))
    cv_idx = [(df[col_group].isin(group_id_list[idx_trn]), df[col_group].isin(group_id_list[idx_val])) for idx_trn, idx_val in kf_list]

    return [(df[idx_trn].index.tolist(), df[idx_val].index.tolist()) for idx_trn, idx_val in cv_idx]

def cal_rmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))


def fit_lgb(x, y, cv, model_params, fit_params, fobj=None, feval=None):
    models = []
    n_records = y.shape[0]
    oof_pred = np.zeros(n_records, dtype=np.float32)
    fold = 0
    data_labels = [re.sub(r'[",\[\]{}:()]', '_', c) for c in x.columns.tolist()]

    model_params.update(deterministic = True)

    for trn_idx, val_idx in cv:

        print(max(trn_idx))
        print(max(val_idx))

        fold += 1
        x_train, x_valid = x.iloc[trn_idx].values, x.iloc[val_idx].values
        y_train, y_valid = np.array(y.iloc[trn_idx]), np.array(y.iloc[val_idx])

        lgb_train = lgb.Dataset(x_train, y_train, feature_name=data_labels)
        lgb_valid = lgb.Dataset(x_valid, y_valid, feature_name=data_labels, reference=lgb_train)

        lgb_model = lgb.train(model_params,
                              train_set=lgb_train,
                              valid_sets=[lgb_train, lgb_valid],
                              fobj=fobj,
                              feval=feval,
                              verbose_eval=fit_params['verbose_eval'],
                              num_boost_round=fit_params['num_boost_rounds'],
                              callbacks=[lgb.early_stopping(fit_params['early_stopping_rounds'])],
                              )

        pred_valid = lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration)
        oof_pred[val_idx] = pred_valid
        models.append(lgb_model)

        print(f' - fold{fold}_RMSE - {cal_rmse(y_valid, pred_valid):4f}')

    print(f' - CV_RMSE - {cal_rmse(oof_pred, np.array(y)):4f}')

    return oof_pred, models


def visualize_importance(models, feat_train_df, file_name):

    feature_importance_df = pd.DataFrame()

    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importance(importance_type='gain')
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df],
                                          axis=0,
                                          ignore_index=True
                                         )

    mean_feature_importance = feature_importance_df.groupby('column').mean().sort_values('feature_importance', ascending=False)
    mean_feature_importance.to_csv(os.path.join(model_path, f'feature_importance_' +config.file_rev +'.csv'), index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:100]

    fig, ax = plt.subplots(figsize=(12, max(8, len(order) * 0.25)))
    sns.boxenplot(data=feature_importance_df,
                  x='feature_importance',
                  y='column',
                  order=order,
                  ax=ax,
                  palette='viridis',
                  orient='h'
                 )
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Importance')
    ax.grid()
    fig.tight_layout()

    return fig, ax


def Traning(df_train, target):
    #cross validation strategy
    kf = KFold(n_splits=config.nfold, shuffle=True, random_state=config.random_seed)
    kf_cv = list(kf.split(df_train))

    lgb_model_params = {'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric' : 'rmse',
                    'seed': config.random_seed,
                    'device':'gpu',
                    'gpu_use_dp' : 'true',

                    'learning_rate': 0.05,
                    'num_leaves': lgbm_numleaves,
                    'verbosity': -1,
                    }

    lgb_fit_params ={'num_boost_rounds': num_boost_rounds_set,
                     'early_stopping_rounds': 100,
                     'verbose_eval': 1000,
                    }

    oof_valid_lgb1, lgb_models1 = fit_lgb(x=df_train, y=target, cv=kf_cv,
                                      model_params=lgb_model_params, fit_params=lgb_fit_params
                                     )

    # modelの保存
    pickle.dump(lgb_models1, open(model_path + modelfilename,'wb'))
    pickle.dump(oof_valid_lgb1, open(model_path + oofvalidname,'wb'))

    print("save model")

    _ = visualize_importance(lgb_models1, df_train, 'lgb_model')

    if config.fVisualize == 1:
        plt.show()

    return

def predict_test(models, df):
    out = np.array([model.predict(df) for model in models])
    out = np.mean(out, axis=0)
    return out

#check distribution of train and test_pred data
def plot_prediction_distribution(y_true, y_pred, y_test):
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.histplot(y_test, label='Test Predict', ax=ax, color='black', stat='density')
    sns.histplot(y_pred, label='Out Of Fold', ax=ax, color='C1', stat='density', alpha=0.5)
    sns.histplot(y_true, label='True Value', ax=ax, color='blue', stat='density', alpha=0.5)
    ax.legend()
    ax.grid()


def Prediction(df_test, target):
    # modelの読み込み
    lgb_models1 = pickle.load(open(model_path + modelfilename,'rb'))
    oof_valid_lgb1 = pickle.load(open(model_path + oofvalidname,'rb'))

    pred_test = predict_test(lgb_models1, df_test)
    pred_test = np.where(pred_test<0,0,pred_test)

    plot_prediction_distribution(target, oof_valid_lgb1, pred_test)

    if config.fVisualize == 1:
        plt.show()

    #Save file
    submit_data = pd.read_csv('data/submit_sample.csv', header=None)
    print('length of submit data')
    print(len(submit_data))
    submit_data.iloc[:, 1] = pred_test
    print(submit_data)

    submit_data.to_csv(submitfilename, index=False, header=None)
    pd.read_csv(submitfilename, header=None)


def corr_column(df, threshold):

    df_corr = df.corr()
    df_corr = abs(df_corr)
    columns = df_corr.columns

    # 対角線の値を0にする
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0

    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None

            # その他との相関の絶対値が大きい方を除去
            if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column

            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)

    return df_corr.columns


class AbstractBaseBlock:
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

class NumericFeatBlock(AbstractBaseBlock):
    def __init__(self, col: str):
        self.col = col

    def fit(self, input_df, y=None):
        pass

    def transform(self, input_df):
        return input_df.loc[:, self.col]

class CategoricalFeatBlock(AbstractBaseBlock):
    def __init__(self, col: str, whole_df = None, threshold=0.001, is_label=True, is_dummy=False):
        self.col = col
        self.whole_df = whole_df
        self.threshold = threshold
        self.is_label = is_label
        self.is_dummy = is_dummy

    def fit(self, input_df, y=None):
        if self.whole_df == None:
            df = input_df.loc[:, self.col]
        else:
            df = self.whole_df.loc[:, self.col]
        vc = df.value_counts(normalize=True).reset_index()
        vc = vc.assign(thresh=lambda d: np.where(d[self.col].values >= self.threshold, 1, 0))\
               .assign(thresh=lambda d: d['thresh'].cumsum() - d['thresh'])
        self.label_dict_ = dict(vc[['index', 'thresh']].values)
        self.label_other_ = np.max(self.label_dict_.values())

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        label_df = pd.DataFrame()
        label_df[f'{self.col}_label_enc'] = np.vectorize(lambda x: self.label_dict_.get(x, self.label_other_))\
                                                        (input_df[self.col].values)
        if self.is_label:
            out_df = pd.concat([out_df, label_df], axis=1)

        if self.is_dummy:
            label_df[f'{self.col}_label_enc'] = label_df[f'{self.col}_label_enc'].astype(object)
            out_df = pd.concat([out_df, pd.get_dummies(label_df)], axis=1)

        return out_df

class DateFeatureBlock(AbstractBaseBlock):
    def __init__(self, is_get_weekday=True):
        self.is_get_weekday = is_get_weekday

    def fit(self, input_df, y=None):
        pass

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = input_df[['year', 'month', 'day']]
        if self.is_get_weekday:
            out_df = out_df.assign(weekday=[x.dayofweek for x in input_df['timestamp'].tolist()])

        return out_df

class AggregateValueBlock(AbstractBaseBlock):
    def __init__(self, key_col, agg_dict, whole_df=None):
        self.key_col = key_col
        self.agg_dict = agg_dict
        self.whole_df = whole_df

    def fit(self, input_df, y=None):
        if self.whole_df == None:
            df = input_df
        else:
            df = self.whole_df
        agg_df = df.groupby(self.key_col).agg(self.agg_dict)
        agg_df.columns = ['_'.join(c) for c in agg_df.columns]
        self.agg_df_ = agg_df.add_prefix('_')\
                             .add_prefix('_'.join(self.key_col))\
                             .add_prefix('agg_').reset_index()

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df = pd.merge(input_df[self.key_col], self.agg_df_, how='left', on=self.key_col)\
                   .drop(self.key_col, axis=1)\
                   .fillna(0)

        return out_df


def get_train_data(input_df, feat_blocks, y=None, fit_df=None):
    if fit_df is None:
        fit_df = input_df.copy()

    for block in feat_blocks:
        block.fit(fit_df, y)

    out = [block.transform(input_df) for block in feat_blocks]
    out = pd.concat(out, axis=1)

    return out

def get_test_data(input_df, feat_blocks):

    out = [block.transform(input_df) for block in feat_blocks]
    out = pd.concat(out, axis=1)

    return out

# main routine
if __name__ == '__main__':
    if config.fmakemap == 1:
        make_map()

    if config.fclustering == 1:
        position_clustering()

    if config.fmakedailyslice == 1:
        make_dailyslice()

    Train_data, Test_data, Cluster_data = read_data()

    #preProcess
    if config.fPreprocess == 1:
        ModifiedTrainData, Target, ModifiedTestData = Preprocessing(Train_data, Test_data, Cluster_data)
    else:
        ModifiedTrainData, Target, ModifiedTestData = read_preprocessed_data()

    if config.fTraining == 1:
        Traning(ModifiedTrainData, Target)

    if config.fPrediction == 1:
        Prediction(ModifiedTestData, Target)



