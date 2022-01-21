# CC-RDG3
協調共進化のグルーピング手法にRDG3を採用した進化計算アルゴリズム

## 1.実行環境
- OS : Windows 10/11, Mac OS
- プログラミング言語 : Python 3.9 以上

### 1-1.必要なモジュールのインストール方法
- `config.py`をメインにして実行すると、モジュールがインストールされます。
- コマンドでのインストール方法は以下のとおりです。
  - Windows
  ```python
  py -3.9 -m pip install -r _config/module.txt
  ```
  - Mac OS
  ```python
  pip3 install -r _config/module.txt
  ```

## 2.コードの構造
コードは「最適化」「ログ・データ処理」「実行」の３つに分けられます。

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
      - 最小化/最大化判定
      - 評価値
    - CCの要素
      - 変数分割(Static Grouping,Random Grouping,RDG3)
      - Context Vectorの補完
      - 評価値配分

#### CCの解の参照方法
- Populationクラスの`pop.x`に「全個体の解」、`pop.f`に「対応する解の評価値」が格納されてます。
- 部分問題の分割数`div`, 個体数`pop`のとき、`div`のリスト（１次元配列）の中に、`pop × subdim`の２次元配列が格納されています。
  - pythonの構造では、 **`list[ndarray[ndarray]]`の３次元配列** になります。

- 例えば、`pop`個目の個体の`div`番目に分割した部分個体の`subdim`次元目にアクセスしたいときは、
```python
pop.x[div][pop, subdim]          # div分割目、pop個体目のsubdim次元の要素
```
と書きます！

- `div`番目に分割した`subdim`次元目の全個体の値を取得したいときは、
```python
❎ pop.x[div][:][subdim]        # = pop.x[div][subdim]
❎ pop.x[div,:,subdim]          # 文法的に❎（ndarray[ndarray[ndarray]]ではない）
⭕ pop.x[div][:,subdim]         # div分割目、全個体のsubdim次元の要素（1次元配列）
```
と書きます。（スライス）

- **注意点**
  - １世代で消費する評価回数は **`max_pop × max_div`回** です！

### 2-2. ログ・データ処理
- 最適化の実験ログや出力データを加工する機能です。
- `DataLogger` / `DataProcessing`などがあります。
- よく使う機能をまとめたラッパークラスが `utils.py` にあるので、そちらを有効利用しましょう！

#### ラッパークラス・関数 (`utils.py`)
- 関数：`log(attrib_name, message)`
  - `print`のラッパー関数が`utils.py`の`log`という関数なので、そちらを使うとログが分かりやすくとれます。
  - `print(f'[関数名] エラー出てます！')` == `log('関数名', 'エラー出てます！')`
  - クラス内では、１つ目の引数に`self`を入れるとクラス名が呼び出し元になります。
  ```python
  from utils import log
  # 関数内でのログの取り方
  log('呼び出し元','ログで出力したいメッセージ')
  # クラス内でのログの取り方
  log(self,'ログで出力したいメッセージ')
  ```
- `Stdio`クラス
  - 標準入出力関係のクラスです
    - 関数：`moveWorkingDirectory()`
      - 作業ディレクトリを移動します。
    - 関数：`makeDirectory(directory, *directories)`
      - フォルダを作ります。
      - コンフリクトがあった場合にはフォルダ名を変更するか、既にあるフォルダを削除するか対話的に決めます。（オプションで無視も可能）
      - パス名が返ってきます。
    - 関数：`getNumberOfFiles(directory, file_template)`
      - テンプレートファイルに一致するファイルが何個あるか数えます。
      - 一致したファイルの個数が返ってきます。
    - 関数：`writeDatabase(df, path_write)`
      - データフレーム`df`を`path_write`にファイル出力します。（`.csv`/`.xlsx`のみ対応）
    - 関数：`readDatabase(path_read)`
      - `path_read`のファイルを読み取り、データフレームにして返します。（`.csv`/`.xlsx`のみ対応）
    - 関数：`saveExperimentalData(path_out, data_dict, file_name)`
      - 辞書型のデータ`data_dict`を`path_out + file_name`のパスに出力します。（`.csv`/`.xlsx`のみ対応）
    - 関数：`drawFigure(~~~)`
      - データを入れるとグラフが書けます。（オプションで様々なカスタマイズが可能）
    - 関数：`outputTracebackError(path_out, error_content)`
      - 例外処理のエラー内容を`path_out`にファイル出力します。



#### ログ
- `DataLogger`は主にログをとるクラスです。（※一部 `Optimizer`内部でログをとっているものもある。）
- データのログやそのグラフをそのまま出力します。
- 機能：
  - 実験環境出力（`config.yml`）
  - 評価値、ベスト評価値の記録

#### データ処理
- `DataProcessing`はデータの加工をするクラスです。
- 機能：
  - 複数試行データの統計処理(五数要約,平均,標準偏差)
  - tSNEによる解のグラフ
  - 多様性尺度の計算、グラフ
  - グルーピングの可視化グラフ

### 2-3. 実行
#### Optimizer/Suboptimizerの単体テスト
- `optimizer.py`をメインにして実行（初期化＆更新）

#### 全体の結合テスト
- `runopt.py`のメイン関数が以下の`runAll()`になっているか確認して、`runopt.py`をメインにして実行
```python
from utils import Stdio
Stdio.moveWorkingDirectory()
runAll()
```

#### パフォーマンステスト
- `runopt.py`のメイン関数が以下の`performanceChecker()`になっているか確認して、`runopt.py`をメインにして実行
```python
from utils import Stdio
Stdio.moveWorkingDirectory()
performanceChecker()
```

#### 全体の実験
- `runopt.py`のメイン関数が以下の`runParallel()`になっているか確認して、`runopt.py`をメインにして実行（並列処理）
##### 試行回数優先（デフォルト）
- １つの関数の試行回数を優先した実行キューを作成
- prob-1,trial-1 -> prob-1,trial-2 -> prob-1,trial-3 ->...
```python
from utils import Stdio
Stdio.moveWorkingDirectory()
runParallel('trial')
# runParallel() でもOK
```
##### 問題優先
- 複数の関数を実行することを優先した実行キューを作成
- prob-1,trial-1 -> prob-2,trial-1 -> prob-3,trial-1 ->...
```python
from utils import Stdio
Stdio.moveWorkingDirectory()
runParallel('problem')
```