# Rubik Simulator Python 
Pythonのゲームエンジン「Ursina」を使用したルービックキューブシミュレーターです。<br>
3x3x3（および2x2x2）のキューブをrubik_3x3x3（およびrubik2x2x2）クラスで**キューブデータを作成**し、RubikCubeCameraクラスで**3D空間に描画・操作**できるほか、OpenCVを使用して<**展開図をリアルタイムで可視化・保存**する機能（rubik_~x~x~クラス）を備えています。<br>
強化学習等にお使いください

![rubik_3x3x3](https://github.com/user-attachments/assets/d4296e42-0b3f-4f5e-93b0-422db3920323)

![rubik_2x2x2](https://github.com/user-attachments/assets/ede62dde-e395-41b4-9386-99455fbc9c62)

## 🚀Features

* **キューブデータの作成**：2次元ndarrayで作成されます。（下詳細）
* **3Dシミュレーション**: Ursina Engineを使用した滑らかな回転アニメーションと操作。
* **2D展開図の可視化**: OpenCVウィンドウにて現在のキューブの状態（展開図）をリアルタイム表示。
* **状態保存**: 現在の状態を画像およびPickleデータとして保存可能。
* **シャッフル機能**: 起動時にランダムにシャッフルされます。
* **論理クラスの分離**: 描画（GUI）とキューブの論理（Logic）が分離されており、強化学習などの環境としても拡張しやすい設計です。

## 📦Requirements

以下のライブラリが必要です。

* ursina
* opencv-python
* numpy

### インストール

```bash
pip install ursina opencv-python numpy
```

## 📁ファイル構成 (File Structure)

* `rubik_3x3x3.py`: 3x3x3 キューブのシミュレーション本体
* `rubik_2x2x2.py`: 2x2x2 キューブのシミュレーション本体
* `savefiles/`: 保存されたキューブの状態（データおよび画像）が格納されるディレクトリ（自動生成）

## 📦使い方 (Usage)

スクリプトを直接実行してください。

```bash
# 3x3x3 キューブの起動
python rubik_3x3x3.py

# 2x2x2 キューブの起動
python rubik_2x2x2.py
```

## 操作方法 (Controls)

### カメラ・システム操作

| 操作 | 説明 |
| :--- | :--- |
| **右クリック + ドラッグ** | カメラの回転（視点移動） |
| **右クリック解除** | カメラ位置のリセット |
| **S キー** | 現在の2Dマップ状態を保存 (`savefiles/`フォルダへ) |
| **R キー** | キューブをリセット（再生成・初期化） |
| **V キー** | 描画の強制リフレッシュ |

### キューブの回転 (テンキー / Numpad)

回転操作はテンキー（Numpad）に割り当てられています。
画面上のヘルプテキストに基づいたキー配置は以下の通りです。

| 面 (Side) | 正回転 (Clockwise) | キー | 逆回転 (Counter-Clockwise) | キー |
| :--- | :--- | :---: | :--- | :---: |
| **Right (右)** | R | `*` | R' | `3` |
| **Left (左)** | L | `2` | L' | `/` |
| **Up (上)** | U | `7` | U' | `-` |
| **Down (下)** | D | `+` | D' | `4` |
| **Front (前)** | F | `6` | F' | `5` |
| **Back (後)** | B | `9` | B' | `8` |

## 技術的詳細 (Technical Details)

### データ構造
キューブの状態は `numpy.ndarray` (9x9グリッド) で管理されています。
これは、各面（U, L, F, R, D, B）を2次元平面上に展開した形式で表現しています。

```
# 9x9 Grid Layout:
#       [U]
#    [L][F][R]
#       [D][B]
```

### クラス設計

* **`rubik_3x3x3` (Logic Class)**
    * 純粋なデータ操作を担当します。
    * `self.cube`: 色情報を持つNumPy配列。
    * 回転メソッド (`R`, `Li`, `U` etc.) による配列の書き換えを行います。
* **`RubikCubeCamera` (GUI Class)**
    * `ursina.Entity` を継承し、3D描画と入力を担当します。
    * 入力での操作は<u>rubik_~x~x~クラスのupdateメゾット</u>により更新されます。
    * Logicクラスの状態を読み取り、小さなCube Entityの集合として3D空間に描画します。
    * 回転アニメーション時は一時的に親Entity (`rotator`) を設定して軸回転させます。

## License

MIT License

## Acknowledgements

This code was developed with the assistance of Google Gemini.