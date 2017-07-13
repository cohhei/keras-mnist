from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# Kerasに含まれるMNISTデータの取得
# 初回はダウンロードが発生するため時間がかかる
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 配列の整形と、色の範囲を0-255 -> 0-1に変換
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

# 正解データを数値からダミー変数の形式に変換
# これは例えば0, 1, 2の3値の分類の正解ラベル5件のデータが以下のような配列になってるとして
#   [0, 1, 2, 1, 0]
# 以下のような形式に変換する
#   [[1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1],
#    [0, 1, 0],
#    [1, 0, 0]]
# 列方向が0, 1, 2、行方向が各データに対応し、元のデータで正解となる部分が1、それ以外が0となるように展開してる
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# ネットワークの定義
# 各層や活性関数に該当するレイヤを順に入れていく
# 作成したあとにmodel.add()で追加することも可能
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

# 損失関数、 最適化アルゴリズムなどを設定しモデルのコンパイルを行う
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 学習処理の実行
model.fit(x_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

# 予測
score = model.evaluate(x_test, y_test, verbose=1)
print('test accuracy : ', score[1])