---
layout: post
category : Machine Learning
tagline: "Supporting tagline"
tags : [CAFFE, Deep Learning]
title : オープンソースDeepLearningフレームワークのCAFFEのLayerを作る
---
{% include JB/setup %}

UC BerkeleyのBVLCを中心にオープンソースで開発しているDeep LearningライブラリのCAFFE。C++/CUDAで書かれているので使い勝手が良く素晴らしいライブラリ。定番のVision系タスクのことは大体できるが、それ以外はまだ開発中って感じ（そもそも開発されるか不明）で、機能拡張したくなる場合があると思う。
そこで、Layerを作るために知っておくべきことをメモ。殴り書き。（誰かが見ると思って書いていないので、上から読んでいっても一回では理解できないと思う。３回くらい読めばわかるかも。文章も適当。）
もし、見て参考にする人がいるとすれば、CAFFEの使い方がある程度わかっている人向けの内容。

# CAFFEの基礎

CAFFEでの学習は基本的に、Netクラス、Solverクラスを使って行われる。どのように使われるかは`tools/caffe.cpp:train()`を見ると割と簡単にわかる。基本的にNetworkの設定やSolverの設定は後述のProtobufによる記述により行われるので、実にシンプルな見た目となっている。ProtobufがわかればCAFFEの構造は大体わかったようなものだ。

## Google Protocol Bufferによる`LayerParameter`等の定義

CAFFEでは各レイヤーの設定やパラメター（学習したWeightなど）を[Google
Protocol
Buffer](https://developers.google.com/protocol-buffers/docs/cpptutorial)
で記述している。これにより、テキストによる設定ファイル
の読み書き、学習したパラメタを保存したり転送したり（バイナリもOK）する
のをすべてGoogle Protocol Bufferにまかせているっぽい。さらに
Protobufは自動で*設定項目に対してインターフェースを備えた*クラスを生成
（Java, Python, C++）してくれて、プログラムから扱いやすいようになって
いる。

protobufの定義はすべて`src/caffe/proto/caffe.proto`に定義されている。
各レイヤーの設定だけではなく、`Blob`や`Datum`などファイルに書き出したいよ
うなものはすべてprotobufとして定義したある。

`LayerParameter`というmessage(*なんて呼ぶのか？*)があり、これがある種
の神クラス的な感じで、すべてのLayerの設定を記述できる形になっている。
変数にConvolution, InnerProduct, Reluなどすべての設定を持っているのだ。

*このParameterプロトコルたちにはパラメタのBlobは含まれないので別途レイヤーのクラスでハンドリングしている。読み書きはBlobProtoでやっていると思うが、どのようなフォーマットで学習結果を書き込んでいるかは要確認。Solverクラスに実装があるはず*

## Blobクラス

`Blob`クラスには`data_`と`diff_`のメンバーがいてそれぞれ
`SyncedMemory`クラスオブジェクト。`SyncedMemory`はCPU/GPUのデータを相
互に参照するタイミングに応じてデータをCPU-->GPUまたはGPU-->CPUに転送し
てくれる便利クラス。data_は変数の値を格納し、diff_はBackpropの時
に、その変数に関する偏微分の値を格納する。このレイヤの処理をy=f(x; h)
（xは入力、yは出力, hはパラメタ）とするときに、x, y, hはdata_に
格納され、（L(x, ..., ; h, ...）を損失関数とする場合に）backpropされて
きた値、dL/dx, dL/dy, dL/dhはそれぞれdiff_に格納されるように使われる。

## Netクラス
NetはNetParameterでインスタンス化される。NetParameterは
lenet_train_test.prototxtのような形で書かれる。NetParameterは次のよう
に定義されている。
{% highlight protobuf %}
message NetParameter {
  optional string name = 1; // consider giving the network a name
  repeated LayerParameter layers = 2; // a bunch of layers.
  // The input blobs to the network.
  repeated string input = 3;
  // The dim of the input blobs. For each input blob there should be four
  // values specifying the num, channels, height and width of the input blob.
  // Thus, there should be a total of (4 * #input) numbers.
  repeated int32 input_dim = 4;
  // Whether the network will force every layer to carry out backward operation.
  // If set False, then whether to carry out backward is determined
  // automatically according to the net structure and learning rates.
  optional bool force_backward = 5 [default = false];
  // The current "state" of the network, including the phase, level, and stage.
  // Some layers may be included/excluded depending on this state and the states
  // specified in the layers' include and exclude fields.
  optional NetState state = 6;
}
{% endhighlight %}

もっとも重要なのはlayersでこれをネットワークの構造に応じてつらつらと書く感じ。input, input_dimは学習時のネットワークをdeployするときに主に使わられるよう（`examples/mnist/lenet.prototxt`参照）。stateはTrainフェーズかTestフェーズかなどを指定する（levelとかもあるけど何に使うのかいまいちまだわかってない）。

`Init`メソッドがコンストラクタで呼ばれネットワークを実際に初期化している。中身は結構複雑でDAGをつなげるところとか、blobの名前から各レイヤーの入出力のblobを作りだしたり、LayerのSetUpを読んだりしている。ちょっと読んだ感じ、FilterNetでlayersごとにメンバであるinclude, excludeオプションをもとに、layresのフィルタリングをしていたり、InsertSplitで２回以上使われているblobをSplitLayerで分岐させたりしている（*なぜかはちゃんと読んでいないので不明。多分同じblobをそのまま異なるLayerの入力にできるようにはなっていないと思われる。BPでdiff_への上書きが発生するから？*）。パラメタシェアリングもここで行っている。BlobのShareDataを使ってblobのdataを同じメモリを参照するようにしている。BlobクラスにはShare{Data,Diff}メソッドが定義されている、*Diffはシェアされないことに注意*これは`Update`の’実装でdiffをownerに集約するところからもわかる。誰がownerかはparam_owners_に記録されている。

`Forward`メソッドを見ると、Forwardはblob(s)を受けるようになっているものとstringでblobvectorを受けるものがあるよう。
ここではnet_input_blobs_に受けとったblobを設定してForwardPrefilledメソッドを呼び出す（学習コード：Solverクラスだとデータはデータレイヤーで勝手に読みだすのでnet_input_blobs_はダミーのblobがセットされる）。
そして中ではForwardFromToが呼ばれlayerごとにForwardが呼ばれる。ここで毎回Reshapeが呼ばれていることに気づいた。つまり、ひとつのバッチ処理ごとにblobのサイズが変わってもちゃんと動くということだ。Layer->Forward(bottoms, tops)はlossを返すようになっており、それが足し込まれて全体のロスを計算している。

`Backward`では単にBackwardFromToで一番後ろから前までを呼ぶ。それだけ。*LossLayerは奇妙なメンバdiff_がいて、Forwardの段階でそこに微分の値が代入されているっぽい。Backwardの実装ではこれをtop diffの代わりに使って、top diffはloss_weightに使われている？*あと、*BackpropされたdiffたちはSolverで使われるのかな*。

`*DebugInfo`メソッドはtop, bottomのblobの統計情報を吐き出してくれて結構役立ちそう。

`ShareTranedLayerWith(other)`は、他のネットとParameterをシェアするときに呼ばれるっぽい。他から自分にシェアする。コピーではなくシェア。パラメタシェアリングに使うっぽい。Shareした場合は、ownerがどこなのかは

`CopyTranedLayerFrom(net_param)`はシェアではなくコピー。pretrainedなモデルを読み込むときには実際にこれが呼ばれている。`tools/caffe.cpp : train()`を見るとわかりやすい。

`Update`メソッドでdiffをつかってweightの更新をしている。w - diffです。diffにはすでにlearninted_rate とweight decayがかかっているらしい。*いつかけたん*。パラメタシェアリングしている場合は、diffをownerに集約する。実際のUpdate: w-diffの処理はBlobクラスに実装されている。*Solverが変わった時とかどうしてんだろ。momentumの時とか。*



## Layerクラス

すべてのLayerの基礎となる親クラス。[CAFFEの本家サイ
ト](http://caffe.berkeleyvision.org/)の[APIドキュメンテーショ
ン](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1Layer.html)
を見る。

コンストラクタは基本的にProtoで定義済みのLayerParameterを受け取る仕様
になっており、\*.prototxtに定義したパラメータを受け取る。また、すでに学
習済みのprotobufのバイナリを設定して受け取る場合は、`Blob`の値をコピー
して受け取っている。Net::Init()やテストコードを見るとインスタンス化された後に呼ばれるのは`SetUp`関数であり、
この中では小クラスで実装されるべき以下のメソッドを呼び出す。

{% highlight c++ %}
CheckBlobCounts(bottom, *top);
LayerSetUp(bottom, top);
Reshape(bottom, top);
SetLossWeights(top);
{% endhighlight %}

#### `CheckBlobCounts`

Layerクラスの実装ではBottom（input）のBlobとTop（output）のBlobのそれ
ぞれの数をメソッド定義する（`{ExactNum,Min,Max}{Bottom,Top}Blobs()`の
どれかの実装がbottom, topについてそれぞれ必要）。

#### `LayerSetUp`

実装クラスでは、パラメタのblobs_を初期化しているのが主な役割っぽい。
（*なぜ、ここでbottomやtopが引数として必要なのか？それはReshapeの役割
ではないか？*）

#### `Reshape`
bottomのShapeに応じて、topのReshapeをするのが主な仕事。Bottomの大きさ
が変わったことで、計算コスト削減目的の一時変数バッファのReshapeなども
行う。

#### `SetLossWeights`
すべてのLayerは各topに対してロスを持つことができる。設定ファイルから読
み込んでLayerParameterオブジェクトに格納されているloss_weightメンバか
らそれぞれのtopについてのlossのweightを読み込んで`loss_`メンバーに書き
込む（*どうやらtopが複数ある場合にはloss_weightを若い方から割り当て
ることしかできないため、もしloss_weightを２個めのtopに設定したい場合は
loss_weight: 0, loss_weight: 1.0のように設定する必要があるよう。,は改
行だと思ってほしい*）。さらに、Lossとして設定されたtop blobはForwardメ
ソッドの中で自動的にscalarになるようにSumを取られて、戻り値として返さ
れる（*この戻り値がどのようにBPで使われてるのか要確認*）。実際にSumの処理は
dot積として実装されており、dot積のための一次変数としてtopのblobのdiff_
が利用されている。*なので、loss_weightを設定したレイヤーのトップは
末端のデータである必要があり次のレイヤーの入力には使えないっぽい（top
のdiffがBPにより書き換わってしまうため）*。

# InterProductレイヤーを例に見てみる
`include/caffe/common_layer.hpp`,
`src/caffe/layers/inner_product_layer.{cpp,cu}`に定義と実装がある。cpp
にはCPU実装が、cuにはCUDA実装が書かれている。
ヘッダの定義を見ると以下のメソッドが定義されている。

* `LayerSetUp`
* `Reshape`
* `type`
* `ExactNumBottomBlobs`
* `ExactNumTopBlobs`
* `Forward_{cpu,gpu}`
* `Backward_{cpu,gpu}`

{% highlight c++ %}
/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_INNER_PRODUCT;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
};
{% endhighlight %}

一個ずつソースを覗いたときのメモ

#### `LayerSetup`
LayerParameterオブジェクトから設定等を読みだして、パラメータの`Blob`オ
ブジェクトを初期化する（ただし、すでに`Blob`が初期化されている場合はや
らない）。`InnerProductLayer`では、Weight行列とバイアス項を初期化する。
また、レイヤーのメンバであるパラメータをBP学習するかのフラグである
`propagate_down_`変数はここですべてTrueで初期化される（*いつFalseに指
定するのかな？*）

#### Reshape

ここでは、bottom (input)のBlobのShapeからTop（Output）のShapeを求めて、
Reshapeさせる。（*そもそも引数のtopとbottomって誰がいつ渡すのか？可変
サイズの入力が入ってくるようなときの想定？*）
また、Bottomの形が変わった時に、計算時間を削減するための一時変数の格納
先（ここでは`bias_multiplier_`）をReshapeしていたりする。

#### Forward_{cpu,gpu}
入出力のbottomからcpu_data()、topから
mutable_cpu_data()からデータを参照し、パラメータの`blobs_`から
cpu_data()でデータをもらう（GPUコードの場合はgpu_data()、ここでデータ
がGPUにあるかCPUにあるかは僕らは気にしなくても`SyncedMemory`クラスが勝
手にデータを転送してくれる。もちろん転送のコストがかかるので、GPUでや
る場合はGPUでの実装をすべてのLayerで定義するほうが良いのは言うまでもな
い。）
あとは単純にBLAS（GPUではCUBLAS）のAPIを呼んで内積を計算している。*なお、
mutable_\*_data()が呼ばれるとどうやら`SyncedMemory`クラスのフラグが立っ
て、中身変更したってなる？か確認する*

#### Backward_{cpu,gpu}
Backwardの処理は、Forwardが呼ばれたあとに格納されたままになっているtop
やbottom等を使いまわす。つまり、Forwardが呼ばれたら入出力のBlobは変わ
らないまま呼ばれるルールのようです。Backwardの結果はBlobのdiff_メンバー
の方に格納される。これがError Backward Propagationの*Error*にあたる。
引数にはtopのBlob、bottomのBlob（diff_が書き換わる）に加え、Bottomに後
方伝搬するかどうかのフラグであるpropagate_downの引数も入力としている
（例えば、ネットワークの入力データなどは伝搬しても意味がないのでFalse
になっている）。

以降はすべてヘッダ実装。

#### type
protobufに定義されているLayerTypeのEnumを返す。

#### ExactNumBottomBlobs, ExactNumTopBlobs
入出力はそれぞれ１つずつしか受け付けない実装になっているので、両方１を
出力する。これをもとにCheckBlobCountsでtop, bottomのBlobの数があってい
るか確認している。柔軟にbottom, topのBlobの数を変更できるような実装に
なっている場合は`{Min,Max}{Top,Bottom}Blobs()`メソッドで最大最小の数を
指定できる。

### テストコードでInnerProductLayerの使われ方を見る
`src/caffe/test/test_inner_product_layer.cpp`の`TestForward`を見ると、
どのようにインスタンス化され、どのような手順で使われるかがわかる。（実
際にユーザーがこれをやることはほぼない。*Netで全部やってくれるはず（要
確認）*）

{% highlight c++ %}
TYPED_TEST(InnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
{% endhighlight %}

最初にLayerParameterオブジェクトを生成してメンバである
InnerProductParameterをprotobufの`mutable_*()`（書き換え可能）のAPIで
とってくる。それに設定（ウェイトの初期化の仕方とか）をsetする。そして、
そのLayerParameterを使って,InnerProductLayerをインスタンス化する。そし
て、予め入力を格納してあるbottomベクターと未初期化（でもよい）topベク
ターを使ってSetUpを呼び、InnerProductLayerのパラメータや一時変数の初期
化を行う。ForwardpropにはFoward関数を呼び出す。結果はtopベクターに格納
されるので、それが間違っていないか確認して終わり。

## Layerクラスのインプリする手順

### 1. レイヤーの設定を定義して、protobufとして定義する。
InnerProductLayerの場合は、まずInnerProductParameter（caffe.proto参照）
を書く。

{% highlight protobuf %}
// Message that stores parameters used by InnerProductLayer
message InnerProductParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms
  optional FillerParameter weight_filler = 3; // The filler for the weight
  optional FillerParameter bias_filler = 4; // The filler for the bias
}
{% endhighlight %}

* 出力数
* バイアス項の有効・無効 
* weightの初期化の仕方
* biasの初期化の仕方

下２つについてはこれらもmessageとして記述がされている。

{% highlight protobuf %}
message FillerParameter {
  // The filler type.
  optional string type = 1 [default = 'constant'];
  optional float value = 2 [default = 0]; // the value in constant filler
  optional float min = 3 [default = 0]; // the min value in uniform filler
  optional float max = 4 [default = 1]; // the max value in uniform filler
  optional float mean = 5 [default = 0]; // the mean value in Gaussian filler
  optional float std = 6 [default = 1]; // the std value in Gaussian filler
  // The expected number of non-zero input weights for a given output in
  // Gaussian filler -- the default -1 means don't perform sparsification.
  optional int32 sparse = 7 [default = -1];
}
{% endhighlight %}

さらに忘れてはならないのが、LayerParameterに新たに`enum LayerType {}`に新たにシンボルと番号（一番最後の数にインクリメントする）を追加。
あと`optinal InnerProductParameter inner_product_param=17;`のようにこれも加える。17の部分LayerParameterの上に書かれているnext available IDの部分を参考に決める。次の人のためにここもインクリメントスル必要あり。

### 2. InnerProductLayerの実装を書く
コンストラクタ、LayerSetUp、Reshape、{Forward,Backward}_{cpu,gpu}、
{Max,Min,ExactNum}{Bottom,Top}Counts、typeなどをオーバーライド実装。

### 3. テストコードを書く。
`src/caffe/test/test_innner_product_layer.cpp`のように正しく動作してい
るかのテストコードを書く。

### 4. layer_factory.cppに`enum LayerType`とLayer実装クラスへのマッピングを記述する。
普通のレイヤーの場合はGetLayer関数に他のものを参考に追加するのみ。適応的に一番ベストなLayerを割り当てたい場合はGet\*Layerを実装する。