# Rinnas Ravencoin (KAWPOW) マイナー

## 概要

このリポジトリは、Ravencoin（RVN）のKAWPOWアルゴリズムに対応したGPU/NPUマイナーです。  
NVIDIA GPU（PyCUDA）とIntel NPU（Intel NPU Acceleration Library）による自動最適化に対応しています。  
RTX 5070ti向けに最適化パラメータが初期設定されています。

## 特徴

- **KAWPOWアルゴリズム**をCUDAカーネルで実装
- **Stratumプロトコル**によるプールマイニング対応
- **Intel NPU Acceleration Library**による自動最適化（NPUがあれば）
- GPU温度・電力・使用率に応じた自動パラメータ調整
- ソロマイニング（テスト用）も可能

## 必要環境

- Windows 10/11
- Python 3.8 以降
- NVIDIA GPU + CUDAドライバ + PyCUDA
- Intel NPU Acceleration Library（NPU最適化を使う場合）

## インストール

1. 必要なPythonパッケージをインストール

```
pip install pycuda numpy
```

2. Intel NPU Acceleration Libraryが必要な場合は、公式手順で導入してください。

3. mining.py を実行できるディレクトリに配置

## 使い方

### プールマイニング

```
python mining.py -w <ウォレットアドレス> -p stratum+tcp://rvn.2miners.com:6060
```

### ソロマイニング（テスト）

```
python mining.py -w <ウォレットアドレス>
```

### オプション

- `-t` / `--threads` : 使用するスレッド数（デフォルト: 1）
- `-d` / `--device`  : 使用するGPUデバイスID（デフォルト: 0）

## 注意・既知の問題

- **DAG生成はダミー実装**です。本番運用にはDAG生成・検証の実装が必要です。
- Intel NPU Acceleration LibraryのAPIは仮想的なものです。実際のAPIに合わせて修正が必要です。
- KAWPOWカーネルは参考実装です。最適化やバグ修正は随時必要です。
- **やる気が途中でなくなったので、細かい部分やエラー処理、DAG生成、NPU最適化モデルの本格実装などは未完成です。**  
  もし続きをやる気が出た方がいたら、ぜひプルリクください。

## 免責

このコードは学習・研究・趣味用途向けです。  
本番運用や資産運用での利用は自己責任でお願いします。

---

**やる気が途中でなくなったので、READMEもこのくらいで許してください。**
