# CC-RDG3
協調共進化のグルーピング手法にRDG3を採用した進化計算アルゴリズム

## 1.実行環境
- OS : Windows 10/11, Mac OS
- プログラミング言語 : Python 3.10 以上

### 1-1.必要なモジュールのインストール方法(Windows)
- Windows
```python
py -3.10 -m pip install -r _config/module.txt
```
- Mac OS
```python
pip install -r _config/module.txt
```

## 2.コードの構造
コードは「最適化」「ログ」「データ処理」「実行」の４つに分けられます。

### 2-1. 最適化
- `Optimizer`(`CCEA`など) / `OptimizerCore` / `Suboptimizer`(`PSO`など) の３種類のコードがあります。
- 協調共進化(CC)の部分

#### コードの要素
概ね下記のような棲み分けです。

- `Optimizer`：最適化全体のコード
  - `SubOptimizer`を継承します！
  - `SubOptimizer`と`OptimizerCore`で定義した関数を使います。
- `SubOptimizer`：最適化手法のコード
  - 最適化手法を実装するコード
    - パラメータの初期化・更新と解集団の初期化・更新の４種類
  - `OptimizerCore`を継承します！
- `OptimizerCore`：共通のコード(フレームワーク)
  - 最適化手法によらない共通要素
    - 最適化の要素
      - 最小化判定、評価、
    - CCの要素
      - グルーピング(静的,RDG3),評価値配分

#### CCの解の参照方法
- Populationクラスの`pop.x`に「全個体の解」、`pop.f`に「対応する解の評価値」が格納されてます。
- 部分問題の分割数`div`, 個体数`pop`のとき、`pop × div`の２次元配列の中に、動的なサイズの`subdim[div]`次元の１次元配列が格納されています。
  - なので、実際は３次元配列（テンソル）です。
- 例えば、`i`個目の個体の`m`番目に分割した部分個体の`k`次元目（次元数`d[m]`）にアクセスしたいときは、
```python
pop.x[n,m]          # d[m]次元の１次元配列
pop.x[n,m][k]        # スカラ
```
と書きます！

- `m`番目の分割の全個体を取りたいときは、
```python
pop.x[:,m]      # pop × d[m] の２次元配列
```
と書きます。（スライス）
- **注意点**
  - １世代で消費する評価回数は **`pop × div`回** です！

### 2-2. ログ
