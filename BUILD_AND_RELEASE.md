# Building the installer & publishing a GitHub release

This guide covers packaging **Transcriber** into a Windows installer and
publishing it as a private GitHub release. Some steps must be run by you
(they need your GitHub login and admin tools); each is spelled out.

---

## 0. One-time: install the build tools

| Tool | Why | Install |
|---|---|---|
| **PyInstaller** | bundles the app into a folder of exe + libs | already installed in the venv (`pip install pyinstaller`) |
| **Inno Setup 6** | wraps that folder into a `Setup.exe` | download from https://jrsoftware.org/isdl.php and install |
| **GitHub CLI (`gh`)** | create the private repo + upload the release | https://cli.github.com/ , then run `gh auth login` |

---

## 1. Build the app (PyInstaller)

From the project folder:

```
.venv\Scripts\python.exe -E -m PyInstaller Transcriber.spec --noconfirm
```

- Output: `dist\Transcriber\Transcriber.exe` (a one-folder build — the exe plus
  all its libraries). This can be **large (2–5 GB)** because of the CUDA/ML
  libraries and bundled TTS voices.
- Test it: double-click `dist\Transcriber\Transcriber.exe`. On first run it
  creates `%APPDATA%\Transcriber\` for config/credentials/logs and downloads the
  Whisper model (~3 GB) to the Hugging Face cache. It needs an NVIDIA GPU + CUDA.

> Note: the Whisper speech model is **not** bundled (it's downloaded on first
> run). TTS voices in `tts_voices\` **are** bundled if present.

---

## 2. Build the installer (Inno Setup)

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\Transcriber.iss
```

- Output: `installer\Output\Transcriber-Setup-1.0.exe`
- This is the single file you distribute. It installs to `Program Files\Transcriber`,
  adds Start-menu (and optional desktop) shortcuts with the app icon, and a
  proper uninstaller. User config/credentials/logs live in
  `%APPDATA%\Transcriber` so the app works without admin after install.

---

## 3. Publish to GitHub (private repo + release)

Run these yourself (needs `gh auth login` done first). From the project folder:

**Create the private repo and push the source:**
```
gh repo create Transcriber --private --source . --remote origin --push
```

**Tag and create the release, attaching the installer:**
```
git tag v1.0
git push origin v1.0
gh release create v1.0 "installer\Output\Transcriber-Setup-1.0.exe" ^
  --title "Transcriber 1.0" ^
  --notes "First release. Live GPU transcription of scanner feeds + PC/app audio, neural text-to-speech read-aloud, and more. Windows + NVIDIA GPU required."
```

That uploads `Transcriber-Setup-1.0.exe` as a downloadable release asset on the
private repo.

> GitHub blocks files over 2 GB as release assets. If the installer exceeds that,
> either (a) split the CUDA libraries out and download them on first run, or
> (b) host the installer elsewhere (e.g. a cloud drive) and link it from the
> release notes. Ask and I can help set either up.

---

## What is / isn't in the repo (secrets)

**Never committed** (in `.gitignore`): `credentials.json` (your Broadcastify
password), `config.json` (your personal feeds), `logs/`, `.venv/`, `tts_voices/`,
`.claude/`. The repo ships `config.example.json` and `credentials.example.json`
templates instead. Verify anytime with:

```
git ls-files | findstr /i "credentials.json config.json"
```
(should show only the `.example.json` files).
