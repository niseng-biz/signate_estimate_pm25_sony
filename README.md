# signate_estimatepm25_sony
ソニーグループ合同 データ分析コンペティション（for Recruiting） 大気中の汚染物質濃度の予測に挑戦しよう！で62thだったコードです。

# 考え方

１日前のpm2.5が時間とともにどこに行くかは地形情報や地上付近の風、上空の風、拡散係数やpm2.5自体の発生など、影響されるものが多すぎるので、基本的に同じ日のデータを使って領域方向の推論で予測をすることを考えました。

# CityのDrop

　目視で明らかにTestデータから遠そうなCityの情報は削っています。　

　'Novosibirsk','Darwin', 'Perth','Ürümqi', 'Naha', 'Calama', 'Sapporo', 'Hegang', 'Bandar Abbas', 'Yazd'を削りました。

# クラスタリング　

　k-meansで40領域に分割しています。pm2.5の推論に対するlgbmの緯度の重要度が低かったため、微妙に緯度方向の重みを減らしてクラスタリングしてみましたが、あまり効果ありませんでした。

# Target Encoding 1

　各都市の周囲に正方領域を切り取り、score = (co_mid * 0.8 + no2_mid*  0.2)の値が一番近い都市を探します。その都市のco_mid、no2_midとpm25_midの値、緯度、経度情報をneighbor情報として追加しています。データが存在しない場合は領域を拡大して探しています。

#  Target Encoding 2

　クラスタリングで求めた40領域を年と月ごと（3年x 12か月で36とおり）にpm25_midの平均値を求めてデータフレームに追加してます。TrainDataのTarget Encordig時はkfoldで4分割して、自分を含むデータが入らないようにしています。Test Dataには単純にTrainDataから求められる平均値を入れています。

Target Encordingした値は平均・最大・最小・標準偏差を求めて特徴量として追加してます。（DIKNさんのコードそのまま）

# データ分割

kfold, 分割数6です。

# model

 LightBGMです。numleaves = 70です。
