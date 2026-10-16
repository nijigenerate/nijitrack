# MediaPipe更新とパッチ管理

管理対象は上流の固定情報、小さなGitパッチ、nijitrackの追加コードです。
フルソースやBazelのキャッシュは管理対象ではありません。

| 場所 | 役割 |
|---|---|
| `../toolchain.lock.json` | MediaPipeアーカイブのコミット、URL、SHA256 |
| `mediapipe/manifest.json` | 対応コミット、パッチの適用順、変更前後のファイルSHA256、置換コードの入力SHA256 |
| `mediapipe/*.patch` | Python初期化の除去、zlibの統一、Halide用依存への変更 |
| `dependencies/manifest.json` | 上流が固定した依存物に対するパッチとファイルSHA256 |
| `dependencies/*.patch` | LLVMのStarlarkオーバーレイ、XNNPACKのC++/CMake生成器、Protobufのzlib依存 |
| `../toolchain/mediapipe/` | 独立したHalide実装・生成器・顔/手アーカイブのビルド定義 |

Windowsの`prepare_mediapipe_halide.ps1`とLinux/macOSの`prepare_mediapipe_halide.sh`はアーカイブのSHA256を検証し、毎回ステージング領域へ
新規展開します。入力ハッシュ、`git apply --check`、出力ハッシュを検証した後で
`build-mediapipe/src`を置き換えます。既存の作業ソースへの手編集は保持しません。
`vendor/sources`の原本へパッチを適用することはありません。

依存物はBazelのfetch後、同じ検証付きパッチを適用します。適用済みの既知ハッシュは
再実行できますが、不明な内容や部分適用はエラーになります。依存物のURL・コミットと
アーカイブSHA256は、固定MediaPipe/TensorFlowのビルド定義とBazelのモジュールロックが
引き続き固定します。依存物を更新した際にも、パッチの再確認が必要です。

## 更新手順

1. `toolchain.lock.json`のMediaPipeコミット、URL、アーカイブSHA256、配置先を更新し、
   セットアップを実行する。ハッシュは実際の公式アーカイブを検証して設定する。
2. 新旧の独立した上流ソースを比較し、必要性を再確認しながら各パッチを新コミットへ
   リベースする。上流で不要になったパッチは削除する。生成済みワークスペースの
   フル差分をそのまま取り込まない。
3. `mediapipe/manifest.json`の対応コミットと、変更前後の実ファイルSHA256を更新する。
   `image_to_tensor_halide.cc`の置換先も入力ハッシュで検証するため、元のインターフェースの
   変更を確認する。依存物が変われば`dependencies`も同様に更新する。
   アプリが使う`vendor/include`のC APIヘッダーも新しい上流と照合し、必要なファイルと
   `assets.lock.json`の版・ハッシュを更新する。prepareは版と実ファイルの一致を検証するため、
   古いヘッダーと新SDKの組み合わせは拒否する。
4. 以下の順序でクリーンなソース生成、パッチ検証、SDK、本体を検証する。

Windows:

```powershell
& native/scripts/prepare_mediapipe_halide.ps1
& native/tests/patch_management.ps1
& native/scripts/build_mediapipe_halide.ps1
& native/tests/patch_management.ps1 -Dependencies
& native/scripts/build_native.ps1
```

Linux/macOS:

```bash
bash native/build.sh prepare
bash native/build.sh sdk
bash native/build.sh build
```

ソース準備と依存物へのパッチ適用は、シェル側でも同じ入力/出力ハッシュを検証する。Windows用の`.ps1`をLinux/macOSで実行する必要はない。
本体のCTestに加え、モデルを使う`tracking-sdk-smoke`で顔・手の検出、HalideのROIテスト、
Windowsは`verify_native_dependencies.ps1`、Linux/macOSは`verify_native_dependencies.sh`で実行時依存も確認する。Linux/macOSのSDKも共通のパッチを使うが、
Windowsの成功を他OSの実測結果として扱わない。Pythonは更新作業・ビルド・テストでも使用しない。

`configuration.json`には上流の版、アーカイブハッシュ、パッチと追加コードの
フィンガープリントを記録します。準備後にこれらが変わるとSDKビルドは再準備を要求します。
`build-mediapipe/src`やキャッシュを配布・コミットする必要はありません。
