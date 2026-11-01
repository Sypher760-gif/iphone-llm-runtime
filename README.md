# iPhone LLM Runtime: LLMBench

LLMBench is an iPhone app that measures three things on your own phone:

1. **Memory bandwidth.** How many GB/s the GPU can read from memory. This sets the speed limit for text generation.
2. **Q4_0 matrix-vector speed.** How close a simple hand-written Metal kernel gets to that limit, at the real layer sizes of Liquid AI's LFM3-1.7B.
3. **llama.cpp baseline.** LFM3-1.7B (Q4_0) run with llama.cpp: a short burst, then 10 minutes of continuous generation with the heat and battery logged every 10 seconds.

The app is built by GitHub Actions on a Mac in the cloud. You do not need a Mac. All steps below are on Windows and the iPhone.

Every number the app shows is measured on your phone by code in this repo. Treat results from different days or different thermal states as not comparable.

## 1. Download the app

Download the newest build from the **Releases** page:
<https://github.com/Sypher760-gif/iphone-llm-runtime/releases/tag/latest>

Click `LLMBench-unsigned.ipa`. This file is not signed, so the iPhone will not run it yet. Step 2 signs it.

## 2. Install it with Sideloadly (Windows)

1. Install Sideloadly from <https://sideloadly.io>.
2. Sideloadly needs Apple's iTunes and iCloud for Windows. Install the versions from apple.com, not the Microsoft Store. The Sideloadly site explains this on its install page.
3. Plug the iPhone into the PC with a cable. Unlock it. Tap **Trust** on the phone if it asks.
4. Open Sideloadly. Drag `LLMBench-unsigned.ipa` onto it, or click the IPA icon and choose it.
5. Type your Apple ID email. A free Apple ID works. No paid developer account is needed. Click **Start** and enter the password when asked.
6. When it finishes, the app is on the phone. It does not open yet.
7. On the iPhone: **Settings > General > VPN & Device Management**. Tap your Apple ID under "Developer App". Tap **Trust**.
8. On the iPhone: **Settings > Privacy & Security > Developer Mode**. Turn it on and restart the phone when asked. (This switch only appears after an app has been installed this way.)
9. Open LLMBench.

**The app stops working after 7 days.** This is a limit of free Apple IDs. To renew it, plug the phone in and press **Start** in Sideloadly again with the same file. Your saved results stay on the phone.

## 3. Get the model file onto the phone

The model is about 1 GB and is not inside the app. Use this exact file:

- Repository: `LiquidAI/LFM3-1.7B-GGUF`
- File: **`LFM3-1.7B-QAD-Q4_0.gguf`** (1,049,615,680 bytes)
- Page: <https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LFM3-1.7B-QAD-Q4_0.gguf>
- Direct download: <https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-QAD-Q4_0.gguf>

This is Liquid AI's Q4_0 build. On the iPhone, open the direct download link in Safari and tap **Download**. The file goes to **Files > Downloads**. Do not leave it only in iCloud Drive. If the file shows a cloud icon, tap it so it downloads to the phone.

## 4. Run the tests

Before you start: plug the phone in or make sure the battery is above 50%, turn off Low Power Mode, and close other apps. Keep LLMBench open and the screen on during every test. The app stops the screen from locking while a test runs.

Write down how the phone felt before you started (cool, warm, just charged). The app records thermal state and battery for you.

1. **Memory bandwidth.** Tap **Run** in section 1. It takes a few seconds. It tests buffers of 64, 128, 256 and 512 MB and shows best and median GB/s.
2. **Q4_0 matrix-vector.** Tap **Run** in section 2. It takes under a minute. It first checks the GPU kernel against a CPU calculation (it shows PASS or FAIL), then times each LFM3 layer size.
3. **llama.cpp.** In section 3, tap **Pick GGUF file** and choose `LFM3-1.7B-QAD-Q4_0.gguf` from Files.
   - **Run burst (100 tokens):** loads the model (first run only, it can take a while), then runs the fixed prompt three times. Each run reports prefill tok/s and decode tok/s.
   - **Run sustained (10 min):** generates continuously for 600 seconds. Every 10 seconds it logs decode tok/s, thermal state, battery and memory. Tap **Stop** to end it early. A stopped run is marked `stopped_early` in the file.
   - Run the burst test first, with the phone cool. Let the phone cool down before the sustained test, or write down that you did not.

If a test fails, the error is shown on screen and saved in the JSON file. Nothing is hidden.

## 5. Send the results back

Each section has a **Share results** button once its test has finished. Tap it and send the file to yourself (AirDrop is not available to Windows, so use Mail, Messages or "Save to Files"). Send all JSON files back. The files are also in **Files > On My iPhone > LLMBench**.

The JSON format has a `schema_version` field. Later versions of the app keep the same field names, so results stay comparable.

## What the app does and does not claim

- Bandwidth is measured with a read-only streaming kernel. It is a ceiling for simple streaming reads, not a promise for real model code.
- The Q4_0 test reads weight copies in rotation (about 192 MB per shape) so the weights do not sit in cache. Results are for single-token matrix-vector products only. No attention, convolution or sampling is included, and the per-token sum in the JSON is labeled as an estimate.
- The layer sizes come from LFM3-1.7B's `config.json` and `modeling_lfm3.py`. The JSON lists them with their source.
- The llama.cpp run uses release **b12024** (commit `ddf48f85c933c1f438212a02eb21e7b001d2d37d`), the prebuilt xcframework from that release, checked by SHA-256 in CI. LFM3 support was added to llama.cpp in pull request #31252 (merged 2026-10-27). Setup: all layers on the GPU, n_ctx 32768, batch 1 for decode, 100 output tokens, a fixed prompt, greedy sampling. If iOS refuses the 32768 context, the app falls back to a smaller one and says so in the file (`n_ctx_used`).
- Peak memory is the process footprint reported by iOS (`phys_footprint`), sampled during the run.
- Free Apple IDs cannot add the "increased memory limit" permission. If iOS ends the app during the model load or a 32768 context, tell us and we will change the setup.

## For developers

- `project.yml` is the XcodeGen spec. `LLMBench.xcodeproj` is generated in CI and is not committed.
- `.github/workflows/build.yml` builds on a free GitHub macOS runner, with code signing off, then packs `Payload/LLMBench.app` into the `.ipa`, uploads it as a workflow artifact, and replaces the rolling `latest` Release.
- Metal kernels: `LLMBench/Shaders.metal`. Swift: `MetalBench.swift` (tests 1 and 2), `LlamaBench.swift` (test 3), `Model.swift` (UI state), `Report.swift` (device info and JSON).
- Nothing in CI needs a secret other than the built-in `GITHUB_TOKEN`.

License: Apache-2.0.
