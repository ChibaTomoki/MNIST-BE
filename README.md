# MNIST サンプル

MNIST を手元で実行するサンプル web アプリのバックエンド(推論のみ)です。

## 動作確認済み環境

- Windows11
- WSL2
- Ubuntu20.04
- Poetry1.4.2
- pyenv2.3.17-5-ga57e0b50

## クローン後初回準備

1. .env ファイルに MONGO_URL を追加し、MongoDB の接続 URL を追加
2. .env ファイルに FE_URL を追加し、フロントエンド の URL を追加
3. `pyenv install 3.11.3` で pyenv に python3.11.3 を追加
4. `pyenv local 3.11.3` でこのプロジェクトの pyenv で python3.11.3 を使うように設定
5. `poetry env use 3.11.3` で poetry で python3.11.3 を使うように設定
6. `poetry install` を実行
7. その他 VSCode の設定

## 実行コマンド

1. `poetry shell`で poetry 環境に入る
2. `poetry run uvicorn main:app --reload` で設定した MongoDB から学習済みモデルを受けとり、API サーバーを起動
