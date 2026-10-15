# nijitrack native experimental

MediaPipe Tasksの**公式C APIを直接呼ぶ**Linux x86-64 / Windows x86-64 / macOS arm64用試作です。
このディレクトリは`nijitrack`リポジトリの`native-experimental`ブランチで管理します。
推論・姿勢計算・視線計算・JSON出力・OSC送信はC++で実装しています。
Pythonインタープリター、Python版MediaPipe、Bazel、OpenCV開発ヘッダーは実行・ビルドに不要です。
動画・カメラのデコードに限りFFmpegコマンドを子プロセスとして利用します。
追加の独自C ABIラッパーはありません。

## 使用している資材

- 配布元: https://github.com/tqphan/libmediapipe
- 配布タグ: `mediapipe-v1.0.0-f9e8d75`
- 配布元コミット: `f9e8d750429322f5e65a22e0d0becb0d3be6e4e6`
- ライブラリのMediaPipeソースSHA: `6d31f1ebc3284db74d211d62bdc4f0a0c29ea120`
- ヘッダー: **上記ソースSHAから取得した未改変の公式ヘッダー**。`master`の宣言を手書きしていません。
- モデル: 既存nijitrackの`face_landmarker_v2_with_blendshapes.task`を共有します。
- 顔画像・動画は同梱・自動取得しません。既存版と同様、通常はカメラを入力に使います。
- 上流SDKのライセンス文書は`vendor/sdk/LICENSE.*`、出典は`vendor/sdk/SOURCE.txt`。
- ハッシュ・シンボル・依存検査は実施していますが、第三者配布物の無害性を保証するものではありません。

## 取得・ビルド・実行

リポジトリのルートから、次の手順で実行します。Linuxの必要条件はglibc 2.39以上、C++20コンパイラ、Make、資材取得用のPython 3です。動画テストにはFFmpegも必要です。
共有ライブラリ、モデル、実行ファイル、ビルド生成物はGitに登録していません。
SDKは固定URLから取得し、展開前後にSHA-256を検証します。
モデルは既存版で用意したリポジトリ直下のファイルを参照します。

```bash
cd native
python3 scripts/fetch_assets.py
make -j2
mkdir -p results
./build/nijitrack-cpp --device 0 --frames 300 --output results/camera.jsonl
```

通常のMake/CMakeビルドではキーボード機能を無効にしています。
キーボード対応ビルドは下の「グローバルキーボード入力」を参照してください。
実行時の共有ライブラリはLinux/macOSでは相対RPATH、Windowsでは実行ファイルと同じディレクトリから参照します。

標準出力にJSONを出す場合は`--output`を省略します。ライブラリのログと集計は標準エラーに出します。

```bash
./build/nijitrack-cpp --list-devices
./build/nijitrack-cpp --device 0 --frames 300 --output results/camera.jsonl
```

カメラは`--device /dev/video0`でも指定できます。このWork環境ではデバイス列挙結果が`[]`であり、
実カメラの撮影、カメラ権限、実機での遅延は未検証です。列挙はsysfsにあるvideoノードを表示します。
撮影可能なノードかの能力照会・永続IDによる選択は今後の実装です。

## OSC / VMC

OSC送信は明示的に`--osc-host`を指定した場合に有効です。

```bash
./build/nijitrack-cpp --device 0 --osc-host 127.0.0.1 --osc-port 39540 --output results/osc.jsonl
```

顔1フレームあたり、Headを1メッセージ、視線4個、MediaPipe表情52個、最後にApplyを送ります。
OSCの数値型は32-bit float、Quaternionの順序は`x,y,z,w`です。

- `/VMC/Ext/Bone/Pos`: `Head`, 鼻の正規化位置3成分、Quaternion4成分
- `/VMC/Ext/Blend/Val`: `GazeRightX`, `GazeRightY`, `GazeLeftX`, `GazeLeftY`
- `/VMC/Ext/Blend/Val`: MediaPipeのカテゴリ名と係数
- `/VMC/Ext/Blend/Apply`: 引数なし

UDPメッセージを個別送信します。ネットワーク上の順序・到達保証はありません。
表情名は現行Python版に合わせたMediaPipe名で、VRMのプリセット名への変換は未実装です。
鼻座標はメートル単位の頭部位置ではありません。顔を失ったフレームはJSONで`face_count=0`、
`pose=null`にし、OSCは送信しません。受信側に残った表情を解除する処理は今後の実装です。

## 出力と座標

JSONLには478点、52表情係数、鼻位置、Quaternion、視線、Euler角、MediaPipeの変換行列を出します。
変換行列は`column-major`と明示しています。

頭部姿勢・視線の算出には、現行Python版と同じランドマーク基準を使っています。
Quaternionは回転行列から直接計算し、前フレームとの符号連続性を確保します。
Euler角は旧版の表示式を引き継ぎ、±180度をunwrapします。
MediaPipeの顔変換行列は別に出力し、既存互換の姿勢へ無断で置き換えていません。

`--flip`は**オーバーレイ画像だけを反転**します。推論、左右の表情名、Yaw、視線は変わりません。
オーバーレイは最初のフレームだけをPPM形式で保存します。常時プレビューGUI、ワイヤー、
固定順の棒グラフ、Dへの組み込みは未実装です。

## グローバルキーボード入力

```bash
./build-keyboard/nijitrack-cpp --device 0 --keyboard --osc-host 127.0.0.1 --osc-port 39540 --output results/camera.jsonl
```

既存Python版の`pynput`で行っていたキー押下・解放のOSC送信を、C++のOS別バックエンドで実装しました。
明示的に`--keyboard`を指定したときだけ有効です。OSC宛先の明示も必要です。
`/VMC/Ext/Key`に型`,isi`で`[1または0, UTF-8文字, Unicodeコードポイント]`を送ります。
特殊キーは送信しません。解放時は、Shiftを先に離しても押下時と同じ文字を送信します。
IMEで確定した文章を取得する機能ではありません。

| OS | バックエンド | 検証 |
|---|---|---|
| Linux/X11 | RECORD拡張 | Xvfb/XTESTで押下・解放・Shift・OSC・推論同時動作を検証済み |
| Windows | WH_KEYBOARD_LL / ToUnicodeEx | GitHub Actionsでビルド。実機でのキー取得はCI対象外 |
| macOS | CGEventTap / Unicode取得 | GitHub Actionsでビルド。入力監視権限を伴う実機動作はCI対象外 |
| Linux/Wayland | 未実装 | XWaylandを全デスクトップのキャプチャと偽らず、明示的なエラーにする |

Linuxのビルドでは通常`libx11-dev`と`libxtst-dev`を用意し、以下で有効にします。
実行時はOSの`libX11.so.6`と`libXtst.so.6`が必要です。キーボード用共有ライブラリはGitに登録していません。
`vendor/keyboard`には前の試作で使用したUbuntuパッケージの出典とライセンスを記録しています。

```bash
cmake -S . -B build-keyboard -DCMAKE_BUILD_TYPE=Release -DNIJITRACK_KEYBOARD=ON
cmake --build build-keyboard -j2
```

Makeなら`make clean`の後で`make KEYBOARD=1`を使います。通常のMake/CMakeの初期設定は
キーボードバックエンド無効であり、映像推論にX11開発依存を必須にしません。
OS別のキーボード部品は単独でもビルドできます。

```bash
cmake -S keyboard -B build-keyboard-module -DNIJITRACK_KEYBOARD=ON
cmake --build build-keyboard-module
```

Windows/macOSでの単独ビルド手順も上記です。macOSは入力監視権限が必要です。
LinuxでXvfbとxdotoolを用意した場合、`python3 tests/keyboard_integration.py --image /path/to/existing-face.jpg`で実入力経路を検証できます。
UNIXソケット作成が禁止された環境では`NIJI_X11_TCP=1`を指定します。この場合、認証cookie付きの
仮想Xサーバーにループバック接続し、テスト終了時に停止します。
カメラがフレームを返さずに停止した場合など、入力ポンプが回らない期間の配信遅延は残ります。

動画は`--fps`（初期値30）に整え、`--width` / `--height`（初期値640 / 480）の範囲へ縮小します。
動画のタイムスタンプは整形後のフレーム番号から、カメラは単調増加時計から設定します。
現時点のカメラ入力は逐次処理で、最新フレームのみを保持する低遅延キューは未実装です。

## ソースからビルド

必要条件: Linux x86-64、glibc 2.39以上、GCC 13相当のC++20コンパイラ、Make。
動画・カメラにはFFmpegをPATHに置きます。CPU版のみです。

```bash
make -j2
```

CMake 3.20以上でも検証済みです。

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j2
cmake --install build-cmake --prefix "$PWD/install"
./install/bin/nijitrack-cpp --model ../face_landmarker_v2_with_blendshapes.task --device 0 --output results/installed.jsonl
```

モデルはアプリの暗黙のオンライン取得を避けるため、既定では作業ディレクトリの
`../face_landmarker_v2_with_blendshapes.task`を参照します。別の場所から起動するときは`--model`で明示します。

## 再検証

Python 3は検証・SDK復元ツールを実行する場合だけ必要です。
トラッキング本体からPythonを起動することはありません。

```bash
python3 scripts/verify_assets.py
```

顔画像・動画を使う検証は、利用者が既に持っている入力を明示した場合だけ実行します。
テスト用入力は配布・取得・コピーしません。画像は顔が1人、動画は全フレームに顔が1人映り、
5フレーム以上あり、開始・終了で顔位置が変わるものを指定してください。

```bash
python3 tests/integration.py --binary build/nijitrack-cpp --image /path/to/existing-face.jpg --video /path/to/existing-video.mp4
python3 tests/keyboard_integration.py --image /path/to/existing-face.jpg
```

指定入力による検証では、ランドマーク・表情・姿勢・OSC・反転・エラー処理を確認します。
出力JSONL・オーバーレイ・ログはGit管理外の`results/`へ保存します。
入力を指定しない場合は、引数エラーで終了し、画像取得を行いません。
既存版との差分は`COMPATIBILITY.md`を参照してください。

## GitHub Actionsによる3 OSビルド

`.github/workflows/native-build.yml`はmain/native-experimentalへのpush、PR、手動起動に対応します。
Linux x86-64（Ubuntu 24.04）、Windows x86-64（VS 2022）、macOS arm64（macOS 15）で
本体とOS別キーボード部品をビルドし、CLI起動・引数エラー・子プロセスの引数と終了状態・インストール後の起動を検証します。
SDKは上流の固定ReleaseをSHA-256で検証して取得します。WindowsではDLLに対応するリンク用の
import libraryを公式C APIの宣言に合わせて生成します。独自のC ABI実装は追加しません。

成果物`nijitrack-native-<OS・architecture>`は実行ファイル、共有ライブラリ、ライセンスを含みます。
WindowsはZIP、Linux/macOSは実行権限を保持したtar.gz形式です。
モデル・顔画像・動画・キャプチャ結果は取得せず、成果物にも含めません。モデルは既存版のものを
`--model`で指定します。動画・カメラ利用時はFFmpegを別途PATHへ配置してください。
Linuxのキーボード機能には実行時もlibX11/libXtstが必要です。

Windows/macOSのローカルビルドは`python scripts/fetch_assets.py`の後、次を使います。

```bash
cmake -S . -B build-cmake -DNIJITRACK_KEYBOARD=ON
cmake --build build-cmake --config Release
cmake --install build-cmake --config Release --prefix install
```

WindowsではVS 2022のx64コンパイラ、macOSではarm64のApple Clangを使います。
Windowsカメラは`--device "カメラのDirectShow名"`、macOSは`--device 0`を使います。
Windowsの数値デバイス指定とWindows/macOSの`--list-devices`は未実装としてエラーを返します。
カメラ・入力監視権限・実キー取得・顔推論の実機検証は、この画像なしCIには含めていません。

## Nightly / 仮リリース

毎日03:17 JST（18:17 UTC）に、mainの`native-nightly.yml`が`native-experimental`の
再利用可能なビルドActionを起動します。GitHub側の混雑により開始が遅れる場合があります。
mainのActions画面から手動実行もできます。

3 OSを同一コミットからビルドし、全ビルド・テストが成功した場合だけ
https://github.com/nijigenerate/nijitrack/releases/tag/native-nightly
のプレリリースを更新します。3 OSのアーカイブと`SHA256SUMS`を配置します。
画像・動画・モデルは含めません。リリース本文にビルド元SHAとAction実行URLを記録します。
`native-nightly`タグは成功したソースコミットへ更新するため、固定版ではありません。
