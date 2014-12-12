---
layout: post
category : Machine Learning
tagline: "Supporting tagline"
tags : [CAFFE, Deep Learning]
title : オープンソースDeepLearningフレームワークのCAFFEのLayerを作る ~ DataLayer編 ~　（書きかけ）
---
{% include JB/setup %}

前回はInnerProductLayerのような一般の処理に関するレイヤーについて書いたので、今回はDataLayerについてメモ。

## 既存のデータレイヤーを見ていく

ざっと見てみると完全に画像入力を前提としているように見受けられる。[API Doc](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1BaseDataLayer.html)から継承関係を見てみると、Layer-->BaseDataLayer-->MemoryDataLayerの筋とLayer-->BaseDataLayer-->BasePrefetchDataLayer-->{DataLayer,ImageDataLayer,WindowDataLayer}等の筋がある（投稿時）。

`BaseDataLayer`はおおよそすべてのDataLayerの親になっていてLayerSetUpメソッドと必要なメンバ変数を定義している。このクラスを継承したレイヤーはTransformParameterをもつことができ、画像のランダムクロップやランダムミラーリング、色変換（スケーリング、差分）を行う。
{% highlight protobuf %}
// Message that stores parameters used to apply transformation
// to the data layer's data
message TransformationParameter {
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  optional float scale = 1 [default = 1];
  // Specify if we want to randomly mirror data.
  optional bool mirror = 2 [default = false];
  // Specify if we would like to randomly crop an image.
  optional uint32 crop_size = 3 [default = 0];
  optional string mean_file = 4;
}
{% endhighlight %}

LayerSetupの中身はこのTransform等の初期化を行う。また、本クラスから継承される子クラスで実装されるルールとなっているDataLayerSetUpを呼び出している。

`MemoryDataLayer`を例に見てみる。（これはDeployの時などに使われるように作られている。あとメモリに乗り切るデータを扱う場合に使える。と思う。ただし、tools/caffe.cppからそのまま呼べる形にはなっていない模様。なぜならデータをセットする必要があるため。）

{% highlight protobuf %}
// Message that stores parameters used by MemoryDataLayer
message MemoryDataParameter {
  optional uint32 batch_size = 1;
  optional uint32 channels = 2;
  optional uint32 height = 3;
  optional uint32 width = 4;
}
{% endhighlight %}

`DataLayerSetUp`メソッドを見ると単純にtop blobのReshapeをしているのが主な処理のよう。このLayerを使うにはForwardするまえにAddDatumVectorでデータをセットしておく必要がある。Forwardの中を見ると、

{% highlight c++ %}
template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
  (*top)[0]->set_cpu_data(data_ + pos_ * this->datum_size_);
  (*top)[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  has_new_data_ = false;
}
{% endhighlight %}

Forwardを呼ぶたびにデータの読み出し位置のカウンタであるpos_が進むようになっており、１週回ると0に戻るように実装してある（全サンプル数がbatch_sizeの整数倍である必要がある）。データレイヤーではこのようなルールで実装する必要があるようだ。

さて、別筋の親クラスである`BasePrefetchingDataLayer`を見ている。データをファイルシステム等から読み出すときにはデータ転送時間や前処理時間がボトルネックになる。そのため、スレッドを立ち上げてネットのForwardが伝搬している間に裏でデータ読込と前処理を行い、オーバーヘッドをなくす。
{% highlight c++ %}
// include/caffe/data_layers.hpp
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~BasePrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
};
{% endhighlight %}
継承元である`InternalThread`クラスではboost:threadをラッパーした`Thread`クラスを用いてThreadをStart／Joinさせるメソッドを提供し、`InternalThreadEntry`メソッドをオーバーライドしてThread内の処理を記述するようになっている。本クラスでは初期化時（LayerSetUp）にThreadをStartしてデータの読み込み（これの子クラスで処理を定義）をさせ、Forward_{cpu,gpu}のメソッドの最初でThreadのJoin（データの読込の完了待ち）をして、読み込んだデータが`prefetch_data_`と`prefetch_label_`に格納されている（子クラスで実装）のでそれをtop blobに伝搬し、終わりにスレッドを生成して、裏でデータ読込をさせる仕組みとなっている。子クラスでは主にDataLayerSetUpとInternalThreadEntryをオーバーライドして定義するのがメインになる。

実際の実装クラスである（多分１番わかりやすい）`ImageDataLayer`を見ていく。
{% highlight c++ %}
// include/caffe/data_layers.hpp

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_IMAGE_DATA;
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
};
{% endhighlight %}
Protobufの定義はこんな感じ
{% highlight protobuf %}
// src/caffe/proto/caffe.proto

// Message that stores parameters used by ImageDataLayer
message ImageDataParameter {
  // Specify the data source.
  optional string source = 1;
  // Specify the batch size.
  optional uint32 batch_size = 4;
  // ... 省略
  // Whether or not ImageLayer should shuffle the list of files at every epoch.
  optional bool shuffle = 8 [default = false];
  // It will also resize images if new_height or new_width are not zero.
  optional uint32 new_height = 9 [default = 0];
  optional uint32 new_width = 10 [default = 0];
  // ... 省略
}
{% endhighlight %}

実装クラスなので、ExactNum{Bottom,Top}Blobsなどの入出力の数を決めるメソッドやtypeなどは必要。ImageDataLayerは次のようなフォーマットのテキストファイルをソースとして画像を読み込む。

    フィイルパス1 ラベル1
	フィイルパス2 ラベル2
    ...
	フィイルパスN ラベルN


`new_{height,width}`をセットすると画像は自動的にそのサイズにリサイズされる。また、`TransformParameter`も設定できるのでそこで階調スケーリング、ランダムミラー・クロップを定義できる。`DataLayerSetUp`メソッドではソースのテキストファイルをパースして、pair(ファイルパス、ラベル)のリスト`lines_`として保持し、最初のデータを少し読みだして、top, prefetchのblobのReshapeを行うなど。実際の読込の処理部は`InternalThreadEntry`メソッドで定義され、実に単純で`batch_size_`分だけlines_から画像（同時にTransform）とラベルを読み込んで`prefetch_data_`、`prefetch_label_`に格納、`line_id_`をインクリメントし、リストをすべて走査したら、`line_id_`を0に戻し、`shuffle=true`の場合`lines_`をシャッフルする。どうやらThreadの後始末をするデストラクタは実装する必要があるみたい。以上。

これで大体仕組みはわかったので、データレイヤの実装手順を実際にやってみる

## データレイヤーを実装する。
ここではLIBSVMのフォーマットファイルからデータを読み込む例を実装してみる。
libsvmのフォーマットファイルはこんな感じ。

    <label> <index1>:<value1> <index2>:<value2> ...

これを読めるようにしたい。

書きかけ


# 気付き・疑問

* どこでtop, bottomのReshapeしてるんだっけ。→Netの設定ファイルでbottom: hoge top: fuga の数で決まる。
数が間違っていないかの確認はクラスメンバメソッドのExactほげとかをみて自動で確認してくれているはず
* 
