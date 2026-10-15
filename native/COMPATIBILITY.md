# 既存Python版との差分

比較元: `nijigenerate/nijitrack`の`nijitrack.py`を読み、試作に必要な機能を再照合しました。

| 既存機能 | C++試作での状態 |
|---|---|
| FaceLandmarker VIDEO / BlendShapes | 実装・実推論検証済み |
| 頭部姿勢・鼻位置・視線・OSC | 実装・OSC受信検証済み。鼻位置は正規化座標 |
| グローバルキー押下・解放 | Linux/X11で実装・検証済み。明示的な`--keyboard`で有効化 |
| キーボードのOS差吸収 | Windows/macOSバックエンドをCIビルド対象に追加。実キー取得は実機未検証 |
| 特殊キー | 旧版もOSC送信せずログ表示のみ。試作では送信せず、ログも出さない |
| `--list-devices` | Linuxのsysfs列挙のみ。旧版のV4L2能力照会・Windows DirectShow・macOS AVFoundation列挙は未移植 |
| `--device` | FFmpeg経由でLinux/V4L2、Windows/DirectShow名指定、macOS/AVFoundationに対応。実機未検証 |
| `--osc-host` / `--osc-port` | 実装。旧版の宛先既定値は維持せず、送信は宛先指定時のみ |
| `--flip` | オーバーレイ表示のみ反転。旧コードの推論前反転による左右・Yawへの影響を避ける |
| `--show` | 常時プレビューウィンドウは未移植 |
| `--show-video` / `--show-wire` | 未移植。現時点は最初のフレームのランドマーク付き画像出力 |
| `--show-text` / `--show-blend` | 未移植。数値はJSONLに出力 |
| v/w/t/bキーで表示切替 | プレビューGUIが未実装のため未移植 |
| 初回モデル取得 | アプリ内自動取得はしない。既存版の`face_landmarker_v2_with_blendshapes.task`を共有 |
| Windows/macOSビルド | プロセス起動・UDP・SDKリンクを移植しCI対象に追加。カメラ列挙とGUIは未移植 |

この試作は既存版の全機能移植が完了した版ではありません。
MediaPipe C APIの直接利用、推論、OSC、およびLinux/X11キーボード経路を確認する試作です。
