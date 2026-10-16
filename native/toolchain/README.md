# SDL3 / Halide tracking build environment

Windows x86-64、Linux x86-64、macOS arm64共通のビルド構成です。実測はWindowsで行っています。モデルの取得やアプリの常駐起動は行いません。
実行・セットアップ・以下の検証ではPythonを使用しません。

## 固定バージョン

| 構成要素 | バージョン / 用途 |
|---|---|
| Visual Studio 2022 | 既存のC++ツール、MSVC 14.44、Windows SDK 10.0.26100.0 |
| CMake / Ninja | VS付属の3.31.6 / 1.12.1で検証 |
| LLVM / clang-cl | 19.1.7。MSVC 14.44のSTLがClang 19以上を要求するため |
| Bazel | MediaPipeの`.bazelversion`に合わせた7.4.1。JDK同梱版 |
| Halide | MediaPipeの`WORKSPACE`に合わせた15.0.1。LLVMを内包した公式バイナリ |
| SDL3 | 3.4.18、公式Windows開発SDK |
| MediaPipe | `6d31f1ebc3284db74d211d62bdc4f0a0c29ea120`のソーススナップショット |

Git for WindowsのBashもBazel用に使用します。システムのPATHやレジストリは変更せず、ツールは`native/vendor/toolchain`、上流原本は`native/vendor/sources`、SDKの作業ソースと生成物は`native/build-mediapipe`に置きます。
CMake・Ninja・MSVC・Windows SDK・Gitは既存インストールを利用します。

Windowsの本体・SDKはclang-cl 19.1.7でビルドします。本体はbuild-sdl-static、SDKはvendor/sdk-halideへ配置します。

## セットアップと検証

WindowsはリポジトリルートからPowerShell 7で実行します。Linux/macOSは`bash native/build.sh all`で実行し、PowerShellは不要です。

```powershell
& native/scripts/setup_native_toolchain.ps1
& native/scripts/prepare_mediapipe_halide.ps1
& native/scripts/build_mediapipe_halide.ps1
& native/scripts/build_native.ps1
```

配置済みアーカイブだけを再利用する場合は、セットアップに`-Offline`を指定します。
セットアップは`toolchain.lock.json`のURL・SHA-256で検証して展開し、環境情報を`vendor/toolchain/environment.json`に保存します。
HalideのハッシュはMediaPipeのビルド定義、SDL3はGitHub公式リリースのdigest、Bazelは公式`.sha256`に基づきます。
LLVMとソーススナップショットは公式HTTPS配布元から取得時のハッシュを固定しています。

現在のPowerShellだけに環境を読み込むには次を実行します。

```powershell
. native/scripts/enter_tracking_environment.ps1
```

共通手順でパッチ・Halide・本体のテストと実行時依存を確認します。
Windowsの追加診断用`test_tracking_toolchain.ps1`は次を確認します（Linux/macOSの共通手順では使用しません）。

1. ClangとWindows SDKのコンパイル・リンク。
2. **固定MediaPipeソースの未改変`rgb_float_generator.cc`**をHalideの`GenGen.cpp`とビルド。
3. 生成した前処理を静的ライブラリにして、RGB→float変換の数値を検証。
4. SDL3の非表示ウィンドウ・ソフトウェア描画。カメラにはアクセスしない。
5. 独立した最小Bazelワークスペースでclang-clを選択し、C++プログラムをビルド・実行。MSVCへのフォールバックはコンパイル時に検出。
6. AOT実行側にOpenCV、Python DLL、HalideコンパイラDLLへの直接依存がないこと。

結果は`results/build-environment/verification.json`、依存DLL一覧は同ディレクトリの`smoke-dependencies.txt`に記録します。
検証環境では`python`・`python3`・`py`を停止用のC++プログラムへ向け、呼び出された場合は失敗させます。
これは絶対パスでの実行まで禁止するOSサンドボックスではありません。上流ビルドを無条件に実行してよいという意味ではありません。

## Python不要のMediaPipe静的ビルド

```powershell
& native/scripts/prepare_mediapipe_halide.ps1
& native/scripts/build_mediapipe_halide.ps1
& native/scripts/build_native.ps1 -Install
```

上流のhermetic Python/pip初期化をC++専用のリポジトリ定義へ置き換えます。
変更は`native/patches`のGitパッチとハッシュ付きマニフェストで管理します。prepareは固定アーカイブから毎回作業ソースを新規展開し、入力・出力のハッシュを検証して適用します。上流原本は変更しません。更新手順は[パッチ管理](../patches/README.md)を参照してください。
LLVMのソースオーバーレイはStarlark、XNNPACKのビルド識別子はC++によるバイト連結とCMakeのSHA256で生成します。
WindowsのProtobufはgzip_streamのzlib依存を補います。Python用のランタイムはC++製停止ガードです。

`mediapipe/face_hand.BUILD.bazel`と`tracking_archive.bzl`が、画像・顔・手のC APIの推移的なC++依存を静的アーカイブへまとめます。
Bazel 7の公開オブジェクト一覧とアーカイバAPIを使用します。コンパイル済みのHalide AOTアーカイブだけは別ファイルです。
アプリでは両方を静的リンクし、処理登録をWHOLE_ARCHIVEで保持します。
MediaPipe DLL・OpenCV・HalideコンパイラDLLは実行時に不要です。

CPU前処理を独立したHalide AOTの任意角度ROI変換へ置き換えます。明示的なGPU無効化フラグは使用しません。上流のWindows条件がGPUを自動的に無効化するため、WindowsでGPUデリゲートを明示したテストはGPU無効エラーになります。
角度0/45/90度、境界のゼロ/端値、テンソルの値域をC++テストで検証します。

Linux x86-64/macOS arm64には`build.sh`と`scripts/*.sh`を用意しています。Bash・curl・tar・jqと既存のCMake/Ninja・Clang・Bazelを使用します。
同じMediaPipe/Halide/SDL3ソースを使用します。実際のビルド検証はWindowsだけで、Linux/macOSの成功は未確認です。
全体の使用手順とSDL3の構成は[native README](../README.md)を参照してください。

Windowsの配布対象は本体・SDKとも静的CRT（/MT）です。本体はCMakeのMSVC_RUNTIME_LIBRARY、SDKはBazelのstatic_link_msvcrtで揃えています。ビルド用Halideコンパイラは配布対象外の共有SDKを使用します。
