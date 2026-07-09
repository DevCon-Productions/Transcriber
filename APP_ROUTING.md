# Transcribing a specific application's audio

The transcriber captures **output devices** (speakers), not individual apps —
Windows doesn't expose reliable per-application capture, and browsers like Chrome
mix all their tabs into one audio stream anyway. But you can get the same result
by **routing each app to its own output device**, then capturing that device.

## The idea

```
App you want (e.g. Fubo in Chrome)  →  output: Realtek Speakers  →  capture "Realtek Speakers"
Another app (e.g. Spotify)          →  output: DacMagic          →  capture "DacMagic"
```

Each app sends its sound to a different speaker/output, and the transcriber
captures whichever output(s) you choose — one per sector.

## How many apps can you separate?

As many as you have **output devices**. On this PC that's a few — e.g. Realtek
Speakers, CA DacMagic, and the monitor's HDMI audio — so you can split that many
apps into separate sectors.

## Step by step

1. **Start the app** playing audio (e.g. open Fubo in Chrome).

2. **Route it to an output device** in Windows:
   - **Settings → System → Sound → Volume mixer** (or right-click the speaker
     icon → "Open volume mixer").
   - Find the app in the list, and set its **Output device** to the one you want
     (e.g. *Realtek Speakers* for app #1, *DacMagic* for app #2).
   - The app's audio now plays only through that device. (To still hear it, that
     device needs to be connected to speakers/headphones; or pick the one you
     normally listen through.)

3. **In the transcriber:** Add a PC-audio sector for that device —
   **+ Custom → Source: pc audio → Speakers: <the device you routed to>**, or
   right-click an existing PC-audio sector → **Change audio source** and pick it.

4. Repeat for the second app on a different device → second sector.

## Verify you picked the right device

In the **Change audio source** dialog, with the app playing, click **🔊 Detect
signal** — the device showing the highest level is the one carrying that app's
audio. Pick it.

## Caveats

- Two tabs/windows of the **same** browser can't be separated — Chrome (and
  similar) mix them into one stream before Windows sees them. Use two *different*
  browsers, each routed to a different output device.
- If you're connected via **Remote Desktop**, local output devices (Realtek,
  DacMagic, HDMI) may be replaced by "Remote Audio" and won't capture the
  physical machine's sound. Do PC-audio capture at the physical console.
- Need more separate outputs than you have hardware for? Install a virtual audio
  cable (see `MULTICHANNEL.md`) to create extra virtual output devices.
