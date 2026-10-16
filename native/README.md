# nijitrack native experimental

カメラ・プレビューにSDL3を使うC++20版です。入力はカメラと静止画像です。
顔・手のMediaPipe C APIと必要な推論コードを静的ライブラリにまとめ、nijitrackへリンクします。
OpenCV、FFmpeg、Pythonインタープリターは使用しません。
Windowsは本体とMediaPipe依存を`/MT`でビルドし、MSVC再頒布ランタイムDLLを同梱・要求しません。実行時にはSDL3の共有ライブラリが必要です。HalideコンパイラとBazelはSDKビルド時だけ使用します。

## Python版との置き換え

`nijitrack`（Windowsは`nijitrack.exe`）と同じ内容の`nijitrack-cpp`を生成します。

```text
nijitrack --list-devices
nijitrack --device 0 --osc-host 127.0.0.1 --osc-port 39540 --flip --show --show-video --show-wire --show-text --show-blend
```

既存のオプション、カメラ時のOSC既定値、推論前の左右反転、表示切替の`v/w/t/b`を維持しています。
詳細は[COMPATIBILITY.md](COMPATIBILITY.md)を参照してください。
カメラ番号はSDL3の列挙順の0始まりです。バックエンドが変わるため、複数カメラがある場合は一覧を確認してください。
モデルは既存の`face_landmarker_v2_with_blendshapes.task`を使用します。実行ファイルの隣・親、作業ディレクトリ・親から探索し、`--model FILE`でも指定できます。
モデルの自動ダウンロードは行いません。

## 共通のビルド方針と手順

Windows x86-64、Linux x86-64、macOS arm64で、上流の固定アーカイブ、パッチ、Halide追加コード、顔・手の静的SDKを共通に使用します。Python・OpenCV・FFmpegはセットアップ・ビルド・実行のいずれでも使用しません。
WindowsはPowerShell、Linux/macOSはBashから、既存のCMake/Ninja・Clang・Bazelを呼び出します。Linux/macOSにPowerShellは不要です。

| OS | 事前に必要な環境 |
|---|---|
| Windows | Visual Studio 2022 C++ツール、Windows SDK、CMake/Ninja、Git for Windows、PowerShell 7 |
| Linux | Bash、Clang、CMake 3.24以上、Ninja、Git、curl、jq、pkg-config、readelf、X11/Xtst・SDLカメラ/音声/表示・EGL/GLESの開発ライブラリ |
| macOS | Bash、Xcodeコマンドラインツール、CMake 3.24以上、Ninja、Git、curl、jq、pkg-config |

リポジトリのルートから実行します。Windows:

```powershell
& native/scripts/setup_native_toolchain.ps1
& native/scripts/prepare_mediapipe_halide.ps1
& native/scripts/build_mediapipe_halide.ps1
& native/scripts/build_native.ps1 -Install
```

Linux/macOS:

```bash
bash native/build.sh all
```

段階別には`bash native/build.sh setup`、`prepare`、`sdk`、`build`、`install`を使います。`all --offline`または`setup --offline`は取得済みアーカイブだけを使います。Linux/macOSではBash・curl・tar・jq・SHA256ツールを使用し、PowerShellを導入しません。
`setup_native_toolchain.ps1 -Offline`は取得済みアーカイブだけを使用します。Bazel依存物も事前にキャッシュされている必要があります。取得物は固定URL・SHA256で検証します。
SDKは全OSで静的リンクし、Windowsでは本体とSDKのMSVCランタイムも`/MT`に揃えます。SDL3は各OS向けの共有ライブラリを同梱します。

依存物は`vendor`、MediaPipeの作業用ソースと生成物は`build-mediapipe`、本体は`build-sdl-static`、配置先は`install`です。
MediaPipeの変更は[パッチ管理](patches/README.md)に分離しています。作業用ソースを直接編集せず、パッチまたは`toolchain/mediapipe`の追加コードを変更してください。C APIヘッダーとの版・内容の一致も検証します。
古いOpenCV版SDKへフォールバックしません。MediaPipeのルートで上流のビルドコマンドを直接実行せず、必ず共通のprepare/buildスクリプトを使ってください。
上流のPython初期化を置き換え、LLVMオーバーレイはStarlark、XNNPACKの識別子はC++のバイト連結とCMakeのSHA256で生成します。Python名の呼び出しはC++製ガードで失敗させます。

本体の共通ビルドスクリプトはCTestと実行時依存検査を実行します。依存検査にはWindowsのLLVM、Linuxのreadelf、macOSのotoolを使用し、OpenCV・FFmpeg・Python・共有MediaPipe・Halideコンパイラへの依存を拒否します。WindowsではMSVCランタイムDLLも拒否します。
Windows用の旧ビルド入口はWindowsスクリプトを呼ぶ互換ラッパーです。MakefileもOSに応じた入口から同じCMakeビルドへ委譲します。

macOSではカメラ/入力監視のOS権限が必要です。Linuxのグローバルキー取得はX11 RECORDで、Wayland単独は未対応です。`--no-keyboard`でキー取得を無効にできます。
実機でのビルド・動作検証はWindowsのみです。Linux/macOSも共通手順を使うCI対象ですが、成功を実測したものとしては扱いません。

## 入力・描画・前処理

SDL3で列挙・取得したカメラ画像をRGBへ変換し、`--width`/`--height`（既定640x480）以内に縮小します。
`--fps`（既定30）はカメラへの要求値です。`--video`による動画ファイル入力は廃止しました。
画像ファイルはMediaPipeの画像C APIで読み込みます。

MediaPipeへの入力前処理はHalide AOTです。傾いた顔や手のROIに必要な任意角度の回転、バイリニア補間、境界処理、テンソル値域の正規化を行います。
HalideのコンパイラDLLはアプリに不要です。アプリはデリゲートを明示せず、C APIの既定値（CPU）を使用します。MediaPipeビルドにGPU無効化フラグは指定しません。ただし固定した上流ソースはWindowsをGPU非対応として自動的に無効化します。WindowsでGPUデリゲートを明示した検証は`ImageCloneCalculator: GPU processing is disabled in build flags`で失敗しました。Linux/macOSのGPU経路は未検証です。
CMakeの`WHOLE_ARCHIVE`でMediaPipeの処理登録オブジェクトを保持します。

取得・推論/OSC送信・表示は標準C++のスレッドと同期機構で分離しています。
カメラとプレビューは最新1フレームを保持します。
SDLのイベント処理・描画はメインスレッドで実行し、停止要求を各スレッドへ伝えてjoinします。
5秒ごとの`performance`ログと終了ログで取得/処理FPS、フレーム間隔、推論時間、キュー待ち時間を確認できます。
描画はSDLの線・点・棒グラフとデバッグ用ASCII文字です。Python版とフォント・外見の完全一致はしません。

```text
nijitrack --image portrait.jpg --repeat 3 --no-osc --output results/image.jsonl
nijitrack --device 0 --frames 300 --no-keyboard --output results/camera.jsonl
```

## 検証と配布

```powershell
ctest --test-dir native/build-mediapipe/halide --output-on-failure
ctest --test-dir native/build-sdl-static --output-on-failure
```

C++テストはCLI・OSC、カメラ用サーフェスのRGB変換/縮小、SDLの実描画とイベント、スレッド停止を検証します。
Halideのテストは任意角度のROIと境界処理・正規化を確認します。合成テスト画像はC++から生成します。

`.github/workflows/native-build.yml`も同じソースビルド手順を使用します。Linux/macOSの実機検証は未実施で、CI結果と区別してください。
CIはツール・取得済みソースと静的SDKを別々にキャッシュします。OS/CPU、ランナーのコンパイラ・システムSDK、固定依存情報、SDKのパッチ・ビルド手順が一致する場合はSDKの再ビルドを省略します。本体のビルド、CTest、顔・手の推論、配布物の起動・依存監査は毎回実行します。
キャッシュは各工程の成功直後に保存するため、その後の本体テストが失敗してもSDKを再利用できます。初回や依存・ツールチェーン変更時にはフルビルドが必要です。
`native-nightly.yml`は毎日03:17 JSTにmainをビルドし、3 OSの検証成功後に`native-nightly`プレリリースを更新します。手動実行もmainに限定します。
mainへの移行時は、`NATIVE_CACHE_SEED_RUN_ID`に指定した成功済みnative-experimental実行の成果物から既存のツール・SDKキャッシュを転送します。キーとSHA256が一致するものを復元して実行先refのキャッシュへ保存するため、同じSDKの再ビルドは不要です。転送用成果物の保持は7日で、実行時には元ブランチと成功状態も確認します。
配布時には`share/nijitrack-cpp`のライセンス・出典を含めます。モデル、画像、キャプチャ結果は配布しません。
動画デコード機能、FFmpegの取得・リンク・同梱はありません。`--show-video`はSDL3のカメラ映像表示を切り替えます。
固定取得情報は`toolchain.lock.json`と`toolchain.unix.lock.json`、MediaPipeの依存物は固定ソース内のWORKSPACEに記録されています。

Windowsでの今回の検証では、6件のアプリテストとHalideのROIテストが通り、FFmpeg除去後の実カメラ30フレームが正常終了しました。
公式テスト画像では顔478点・52表情・変換行列と両手21点・世界座標・左右判定を各3フレーム確認しています。
アプリ本体でも顔画像の反転とSDL全レイヤー描画を3フレーム確認しました。
CIは固定URL・SHA256の公式モデルと画像を`results/ci-fixtures`へ取得し、顔・手の推論を検証します。通常のセットアップでは取得せず、配布物にも含めません。
明示的なモデル・画像を使う追加検証は次で実行できます。

```text
tracking-sdk-smoke FACE_MODEL HAND_MODEL FACE_IMAGE HAND_IMAGE [--gpu]
```

静的SDKのシンボル一覧にOpenCV/Pythonがなく、配置物のDLLインポート表にFFmpeg/OpenCV/Pythonがないことを確認しています。
記録は`results/build-environment/static-validation.json`、`runtime-dependencies.txt`、`tracking-static-symbols.txt`です。

旧OpenCV/FFmpeg資材と旧ビルドディレクトリは整理済みです。現行の本体・MediaPipeビルド領域、静的SDK、モデル、再ビルド用ツールと固定ソースは保持します。
