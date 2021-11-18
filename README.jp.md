# PyTorchとOpenVINOによる推論の最適化

## 概要

ディープラーニングにおける推論は、多くのepochを繰り返す学習と異なり、学習済みのモデルを用いて、入力を一回のみ処理する。その際、モデルの重みは更新しないため、計算にかかる負荷は学習に比べて軽量である。

一方で、推論が行われるハードウェア環境は様々であり、GPUを代表とするアクセラレータや、高速なプロセッサが期待できないエッジコンピューティングも考えられる。さらに、カメラ入力に対する画像認識などではリアルタイム処理が求められることもあり、推論に特有の高速化手法があれば有用である。

そのような推論のための高速化手法のうち、OpenVINOはインテルが主に同社のプロセッサ製品向けに提供しているツールキットである。OpenVINOの高速化は、モデルの最適化と量子化の二つの部分から成る。モデル最適化は推論で不要となる処理を省略することでモデルを効率化する。量子化は浮動小数点表現の重みを8bitの整数表現に変換することで高速化を行う。モデル最適化と異なり、量子化では推論の結果が変わるので、精度検証が必要である。

このレポジトリでは、ユーザ定義モデルおよびModel Zooの登録モデルによる画像認識を例として、OpenVINOを利用した推論の高速化の例を示す。ユーザ定義モデルはmodel.pyに定義されたConvNetであり、Model Zooの登録モデルからはResNet-50を例とした。データセットにはそれぞれMNISTおよびCIFAR10を利用している。

## 実行環境

実行環境の例として、Python3のvenvを用いた設定を以下に示す。

```bash
python3 -m venv openvino
source openvino/bin/activate
pip install --upgrade pip
pip install openvino_dev torchvision onnx
```

後で紹介する`mnist_convert.sh`はこの環境が存在しない場合、自動的に環境を構築する。

## 学習モデルの構築とモデルの最適化および量子化

### 学習済みモデルの準備

Model ZooのResNet-50に対しては学習済みモデルがダウンロードできるが、ユーザ定義モデルでは先に学習を行う必要がある。次のスクリプトはMNISTデータセットに対してConvNetによる学習を行い、学習済みモデルを保存する。

OpenVINOはPyTorch標準のモデル形式をそのまま変換できないため、PyTorch形式に加えてポータブルなONNX形式でも保存する。この際、最後のバッチ数を含む次元でモデルが保存されるため、`drop_last=True`を指定して余りのデータを捨てている。学習済みモデルのデフォルトの保存先はmodel/convnet.pthおよびmodel/convnet.onnxである。ファイル名はname引数で変更可能である。

```bash
python mnist.py --epochs 100
```

ResNet-50では次のコマンドでモデルをダウンロードする。上記の環境が構築されている場合、モデルをダウンロードするスクリプトはvenvの実行パスに追加されている。学習済みモデルの保存先は`public/resnet-50-pytorch/resnet50-19c8e357.pth`となる。

```bash
omz_downloader --name resnet-50-pytorch
```

学習済みモデルが用意できた時点で、PyTorchによる推論が実行可能となる。`mode`引数で`pytorch`を指定するが、デフォルトは`pytorch`であるため、引数なしで実行してもよい。

```bash
python mnist_infer.py --mode pytorch
python resnet-50_infer.py
```

### モデルの最適化

次にOpenVINOによるモデル最適化を行う。

上記の実行環境を構築している場合、モデル最適化のためのスクリプトmoもvenvの実行パスに追加されている。

```bash
mo --input_model model/convnet.onnx --output_dir model
```

上記のスクリプトの結果、modelディレクトリに`convnet.xml`, `convnet.bin`, `convnet.mapping`のファイルが生成されている。

Model ZooのResNet-50に関しては、同様にvenvの実行パスにある変換スクリプトから、

```bash
omz_converter --name resnet-50-pytorch
```

を実行することで、`public/resnet-50-pytorch`以下にONNX形式、FP32, FP16, FP16-INT8のモデルが変換される。

最適化モデルによる推論の実行は、推論スクリプトの`mode`引数を`fp32`として実行する。

```bash
python mnist_infer.py --mode fp32
python resnet-50_infer.py --mode fp32
```

mode引数はいまのところpytorch, fp32, int8のみ対応している。

### モデルの量子化

最後に最適化モデルを入力として量子化を行う。量子化は最適化までのプロセスと異なり、精度が変わるため、変換時に検証データを用いてcalibrationを行う必要がある。MNISTでは次のスクリプトでcalibrationのためのデータを抽出する。データはデフォルトでは`data/MNIST`以下に保存される。

```bash
python extract_images.py
```

さらに、検証データとその教師ラベルの対応を量子化スクリプトに与えるため、次のアノテーションスクリプトを用いてアノテーションを生成する。ここで、`mnist_csv`のアノテーション定義はOpenVINOで事前に定義されている。ユーザ定義のアノテーションを作成することも可能である（https://github.com/taneishi/CheXNetを参照）。作成されたアノテーションは`annotation`ディレクトリ以下に、`mnist_csv.pickle`, `mnist_csv.json`として保存される。

```bash
mkdir -p annotation
convert_annotation mnist_csv --annotation_file data/MNIST/val.txt -o annotation
```

OpenVINOによる量子化では、最適化モデルの位置、量子化の手法、calibrationのためのテストデータの位置、指標等をjsonあるいはyaml形式の設定ファイルで指定する。ConvNetのための設定ファイルはあらかじめconfig以下のconvnet.yaml, pot.yamlに保存してある。設定ファイルまで揃ったら、量子化スクリプト`pot`を実行する。

```python
pot -c config/pot.yaml
```

このスクリプトが成功すると、`results/convnet_DefaultQuantization/[date time]/optimized`以下に、量子化後のモデル`convnet.xml`, `convnet.bin`, `convnet.mapping`が生成される。[date time]は実行時の日時から設定される。

生成された量子化後のモデルファイルを`model/INT8`以下に移し、推論スクリプトの`mode`引数を`int8`として実行することで、量子化モデルによる推論が実行可能である。

```bash
python mnist_infer.py --mode int8
```

ここまでのConvNetに関する操作は、次のスクリプトでまとめて実行できる。

```bash
bash mnist_convert.sh
```

## TODO

- [ ] ResNet-50の量子化
