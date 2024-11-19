## 降水量データの超解像
このレポジトリは以下のレポジトリをフォークして作成しました．以下のレポジトリのREADMEを参照してください。

[BasicSR](https://github.com/XPixelGroup/BasicSR)

## How to

### 訓練の実行

1. データセットの準備

  - 高解像度画像と低解像度画像のペアを用意し，それぞれ異なるディレクトリに保存してください．ディレクトリ構造の例は以下の通りです．
    ```
    dataset
    ├── HR
    │   ├── train
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   └── ...
    │   └── val
    │       ├── 0001.npy
    │       ├── 0002.npy
    │       └── ...
    └── LR
        ├── train
        │   ├── 0001.npy
        │   ├── 0002.npy
        │   └── ...
        └── val
            ├── 0001.npy
            ├── 0002.npy
            └── ...

    ```
    `/data02/rito/workspace/SR/itoru/Data` にデータセットを保存していますので確認ください


2. yamlファイルの作成

- 超解像モデルのパラメータ・データセット・損失関数・最適化手法などの設定を行うためのyamlファイルの作成を行ってください．
- `BasicSR_Rain/optins/train/SwinIR/train_SwinIR_SRx4_scratch.yml` に設定ファイルを保存していますので確認ください
- 以下に設定ファイルの例を示します．
  ```yaml
  # general settings
  name: train_SwinIR_SRx4_scratch_rain_d4PDF
  model_type: SwinIRModel
  scale: 4
  num_gpu: auto
  manual_seed: 0

  # dataset and data loader settings
  datasets:
    train:
      name: Rain-Train
      type: PairedRainDataset
      dataroot_gt: ../../Data/HR/d4PDF/train
      dataroot_lq: ../../Data/LR/d4PDF/train
      filename_tmpl: '{}'
      io_backend:
        type: disk

      gt_size: 64
      use_hflip: true
      use_rot: true

      # data loader
      num_worker_per_gpu: 6
      batch_size_per_gpu: 32
      dataset_enlarge_ratio: 1
      prefetch_mode: ~

      normalize:
        upper_bound: 20

  ```

3. 訓練の実行
- BasicSR_Rain/basicsr/train.py のオプションに上記のyamlファイルを指定して訓練を実行してください．
- 以下に訓練の実行例を示します．
  ```bash
  python3 train.py -opt ../options/train/SwinIR/train_SwinIR_SRx4_scratch.yml
  ```

上記の手順で訓練を実行すると，BasicSR_Rain/experiments/以下に訓練結果が保存されます．modelsディレクトリにモデルの重みが，training_statusにはステータスが保存されます．（訓練のステータスを利用して途中から訓練を再開することも可能です．）


### テストの実行
1. テストデータの準備
- 訓練データと同様に高解像度画像と低解像度画像のペアを用意してください．

2. yamlファイルの作成
- テスト時の設定を行うためのyamlファイルの作成を行ってください．
- `BasicSR_Rain/optins/test/SwinIR/test_swinir_x4.yml` に設定ファイルを保存していますので確認ください
- このファイルに学習したモデルの重みを指定してください．

