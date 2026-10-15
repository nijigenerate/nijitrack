# nijitrack native experimental

MediaPipe Tasksの**公式C APIを直接呼ぶ**Linux x86-64 / Windows x86-64 / macOS arm64用試作です。
このディレクトリは`nijitrack`リポジトリの`native-experimental`ブランチで管理します。
推論・姿勢計算・視線計算・JSON出力・OSC送信はC++で実装しています。
Pythonインタープリター、Python版MediaPipe、Bazelは実行・ビルドに不要です。OpenCVのC++開発パッケージはビルドに必要です。
全OSでカメラ・動画をOpenCVのVideoCapture、プレビューをOpenCVのHighGUIで扱います。ffmpeg.exeなどの外部デコードプロセスは起動しません。
映像取得はOpenCV版を採用し、バックエンド選択は全OS共通の`CAP_ANY`に委ねます。WindowsのMSMF・DirectShowとFFmpeg取得の比較では、DirectShowやFFmpegへの切替による安定性の明確な改善は確認できませんでした。
追加の独自C ABIラッパーはありません。

## Python版からの置き換え

全OSで`nijitrack`（Windowsは`nijitrack.exe`）を生成します。`nijitrack-cpp`も同じ実装の別名として残します。
PATH上のPython版をnative版に置き換えると、既存のオプションをそのまま使えます。

```bash
nijitrack --list-devices
nijitrack --device 0 --osc-host 127.0.0.1 --osc-port 39540 --flip --show --show-video --show-wire --show-text --show-blend
```

- カメラ使用時はOSCを既定で`127.0.0.1:39540`へ送り、グローバルキーボード入力も有効です。
- `--flip`はPython版と同じく推論前に画像を反転します。推論結果とOSCにも反映します。
- `--show`でプレビューを開き、`--show-video`/`--show-wire`/`--show-text`/`--show-blend`で初期表示を選びます。`v/w/t/b`で表示を切り替え、ウィンドウを閉じると終了します。
- 全OSで`--device N`に列挙の整数IDを使います。macOSはPython版と同じuniqueID順です。
- カメラ使用時のJSON出力は既定で無効です。追加の`--json`または`--output FILE`で有効にできます。`--no-osc`/`--no-keyboard`で既定の送信・入力を無効にできます。
- モデルは実行ファイルの隣とその親、作業ディレクトリとその親から探します。配布時はモデルを実行ファイルの隣に置くか`--model FILE`を指定します。自動取得はしません。

プレビューは全OS共通のOpenCV HighGUIで実装しています。
LinuxはOpenCVの開発パッケージとGUIバックエンドが必要です。グローバルキー取得にはX11/Xtstも使います。Wayland単独のグローバルキー取得は
既存バックエンドと同じく未対応で、`--no-keyboard`かX11セッションを使います。
`NIJITRACK_KEYBOARD=OFF`でビルドした場合も、カメラには`--no-keyboard`が必要です。
macOSのカメラ・キーボードにはOSのカメラ/入力監視権限が必要です。

今回の変更はWindowsでビルド、引数・既定値・反転・合成画像の描画・表示切替・終了・デバイス列挙を検証しています。
Linux/macOSの実装は追加済みですが、この環境ではビルド・実機検証していません。実カメラによる顔推論も今回未検証です。

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

リポジトリのルートから、次の手順で実行します。CMake 3.20以上、C++20コンパイラ、OpenCV 4のcore/imgproc/videoio/highgui開発パッケージが必要です。Linuxはglibc 2.39以上とキーボード用のX11/Xtst開発パッケージも必要です。Windowsは資材準備時に一致するOpenCV 4.14.0公式開発パッケージを取得します。
共有ライブラリ、モデル、実行ファイル、ビルド生成物はGitに登録していません。
SDKは固定URLから取得し、展開前後にSHA-256を検証します。
モデルは既存版で用意したリポジトリ直下のファイルを参照します。

```bash
cd native
cmake -P scripts/setup_native.cmake
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --config Release --parallel 2
mkdir -p results
./build-cmake/nijitrack --device 0 --frames 300 --output results/camera.jsonl
```

通常のMake/CMakeビルドではキーボード機能を有効にしています。
キーボード対応ビルドは下の「グローバルキーボード入力」を参照してください。
実行時の共有ライブラリはLinux/macOSでは相対RPATH、Windowsでは実行ファイルと同じディレクトリから参照します。

標準出力にJSONを出す場合は`--output`を省略します。ライブラリのログと集計は標準エラーに出します。

```bash
./build-cmake/nijitrack --list-devices
./build-cmake/nijitrack --device 0 --frames 300 --output results/camera.jsonl
```

Linuxのカメラは`--device /dev/video0`でも指定できます。列挙時はV4L2能力を照会し、
映像キャプチャ可能なノードだけを表示します。実カメラの撮影と遅延は今回未検証です。

## OSC / VMC

カメラ利用時はOSC送信がPython版と同じく既定で有効です。画像・動画モードは`--osc-host`を指定した場合に有効です。

```bash
./build-cmake/nijitrack --device 0 --osc-host 127.0.0.1 --osc-port 39540 --output results/osc.jsonl
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

`--flip`は**推論前の画像を反転**します。Python版と同じく左右・Yaw・視線に影響します。
オーバーレイは最初のフレームだけをPPM形式で保存します。常時プレビューは`--show`を使います。
Dへの組み込みは未実装です。

## グローバルキーボード入力

```bash
./build-keyboard/nijitrack-cpp --device 0 --keyboard --osc-host 127.0.0.1 --osc-port 39540 --output results/camera.jsonl
```

既存Python版の`pynput`で行っていたキー押下・解放のOSC送信を、C++のOS別バックエンドで実装しました。
カメラ利用時は既定で有効です。画像・動画モードでは`--keyboard`と`--osc-host`を明示します。
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
キーボードバックエンド有効です。LinuxではプレビューにもX11開発依存が必要です。
OS別のキーボード部品は単独でもビルドできます。

```bash
cmake -S keyboard -B build-keyboard-module -DNIJITRACK_KEYBOARD=ON
cmake --build build-keyboard-module
```

Windows/macOSでの単独ビルド手順も上記です。macOSは入力監視権限が必要です。
通常の検証はC++のCTestで行います。Linuxのグローバルキー取得はX11/Xtstを使うOS別アダプタです。
UNIXソケット作成が禁止された環境では`NIJI_X11_TCP=1`を指定します。この場合、認証cookie付きの
仮想Xサーバーにループバック接続し、テスト終了時に停止します。
キーボード取得とHighGUIのイベント処理はメインスレッドで行い、映像取得・推論待ちから独立して動作します。

カメラはOpenCVに`--fps`（初期値30）とキャプチャ寸法を要求します。カメラ・動画フレームは`--width` / `--height`（初期値640 / 480）の範囲へ縮小します。動画のフレームは複製せず順に読み込みます。
動画のタイムスタンプは整形後のフレーム番号から、カメラは単調増加時計から設定します。
全OSとも標準C++のstd::thread、mutex、condition_variableで、VideoCaptureによる取得、推論・OSC送信、表示・キーボード処理を分離します。カメラは未処理の最新1フレームだけを保持して遅延の蓄積を防ぎ、動画ファイルは処理完了を待って全フレームを順番に渡します。表示用フレームも最新1枚だけを保持します。停止時は待機を解除して各スレッドをjoinします。ただし進行中のOpenCV読み込みは完了を待ちます。
終了ログには処理FPS、取得数、破棄数、取得から処理完了までの平均時間も出力します。取得と表示のコードにOS固有APIはありません。デバイス名の列挙、実行ファイルのパス取得、OSCソケット、グローバルキー取得は共通C++/OpenCV APIで扱えないためOSアダプタに分離しています。

実行中は5秒ごとに標準エラーへ`performance`ログを出力します。`capture_fps`は取得、`processed_fps`は推論処理、`face_fps`は顔を検出したフレームの実測レートです。`capture_interval`と`output_interval`はフレーム受信間隔と推論・OSC処理完了間隔、`read`はOpenCVの読み込み時間、`inference`は推論時間、`queue`はフレーム受信から推論側で取り出すまで、`frame_age`は受信からそのフレームの処理完了までの時間です。それぞれ平均・標準偏差・最大値をミリ秒で記録します。センサー内部や受信アプリ側の遅延は含みません。初回区間にはカメラ初期化時間も含まれます。
カメラ開始時はバックエンド名、要求FPS、OpenCVが報告するFPS・寸法・FOURCCも記録します。報告FPSは実測値とは限りません。

## ソースからビルド

必要条件: Linux x86-64、glibc 2.39以上、GCC 13相当のC++20コンパイラ、Make。
カメラ・動画にはOpenCVのvideoioバックエンドを使います。外部FFmpeg実行ファイルは不要です。Windowsの動画デコーダDLLはOpenCV公式配布物から同梱します。CPU版のみです。

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --config Release --parallel 2
```

CMake 3.20以上でも検証済みです。

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-ccmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --config Release --parallel 2
cmake --install build-cmake --prefix "$PWD/install"
./install/bin/nijitrack-cpp --model ../face_landmarker_v2_with_blendshapes.task --device 0 --output results/installed.jsonl
```

モデルは自動取得せず、実行ファイルの隣やリポジトリルートなどから
`face_landmarker_v2_with_blendshapes.task`を探します。別のモデルは`--model`で明示します。

## 再検証

資材準備とテストにPythonを使用しません。

```bash
cmake -P scripts/setup_native.cmake
cmake --build build-cmake --config Release --parallel 2
ctest --test-dir build-cmake -C Release --output-on-failure
```

C++テストはOpenCVで合成動画を生成・読み込みし、BGR→RGB変換、縮小、EOF、ワイヤー・文字・棒グラフの描画、CLIとOSCを確認します。GUIを開かず、カメラや顔画像にはアクセスしません。
既存のPythonベースの追加テストスクリプトはこのビルド・CI手順では実行しません。

## GitHub Actionsによる3 OSビルド

`.github/workflows/native-build.yml`はmain/native-experimentalへのpush、PR、手動起動に対応します。
Linux x86-64（Ubuntu 24.04）、Windows x86-64（VS 2022）、macOS arm64（macOS 15）で
本体とOS別キーボード部品をビルドし、CLI起動・引数エラー・子プロセスの引数と終了状態・インストール後の起動を検証します。
SDKはCMakeスクリプトで上流の固定ReleaseをSHA-256検証して取得します。Linuxはlibopencv-dev、macOSはHomebrew OpenCVを利用します。WindowsではDLLに対応するリンク用の
import libraryを公式C APIの宣言に合わせて生成します。独自のC ABI実装は追加しません。

成果物`nijitrack-native-<OS・architecture>`は実行ファイル、共有ライブラリ、ライセンスを含みます。
WindowsはZIP、Linux/macOSは実行権限を保持したtar.gz形式です。
モデル・顔画像・動画・キャプチャ結果は取得せず、成果物にも含めません。モデルは既存版のものを
`--model`で指定します。Linux/macOSはOpenCVの実行時ライブラリも必要です。WindowsのOpenCV動画デコーダDLLは同梱します。
Linuxのキーボード機能には実行時もlibX11/libXtstが必要です。

Windows/macOSのローカルビルドは`cmake -P scripts/setup_native.cmake`の後、次を使います。

```bash
cmake -S . -B build-cmake -DNIJITRACK_KEYBOARD=ON
cmake --build build-cmake --config Release
cmake --install build-cmake --config Release --prefix install
```

WindowsではVS 2022のx64コンパイラ、macOSではarm64のApple Clangを使います。
全OSで`--list-devices`と`--device 0`を使います。
WindowsはDirectShow名、macOSはuniqueIDや名前も追加で指定できます。
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
