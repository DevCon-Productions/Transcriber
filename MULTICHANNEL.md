# Transcribing multiple channels of the same app separately

The built-in **PC audio** source captures "Stereo Mix" — the single mixed output
of your PC. That's perfect for transcribing one thing you're watching, but it
**cannot separate two streams of the same app** (e.g. two Fubo channels in
Chrome), because Chrome mixes all its tabs/windows into one audio stream before
Windows can tell them apart.

To transcribe two same-type channels as **separate sectors**, you need to give
each one its own capturable audio device. Here's the reliable way.

## The idea

```
Fubo channel A  →  Browser 1  →  Virtual Cable A  →  app captures "Cable A"  → sector A
Fubo channel B  →  Browser 2  →  Virtual Cable B  →  app captures "Cable B"  → sector B
```

Each browser sends its sound to a different virtual "speaker," and each virtual
speaker shows up as its own recording device the app can capture independently.

## One-time setup

1. **Install a virtual audio cable tool** (free options):
   - [VB-CABLE](https://vb-audio.com/Cable/) (one cable; install twice via the
     A+B "VoiceMeeter Potato" bundle, or use the multi-cable VB-Audio "Cable
     A/B" pack), or
   - [VoiceMeeter](https://vb-audio.com/Voicemeeter/) for more routing control.

   After install you'll have extra playback devices like **"CABLE-A Input"** and
   **"CABLE-B Input"**, each paired with a matching **recording** device.

2. **Use two different browsers** for the two channels — e.g. Chrome for one,
   Edge (or Firefox) for the other. (Two windows of the *same* browser won't
   separate; different browsers are different processes.)

## Per-session steps

3. Open channel A in Browser 1, channel B in Browser 2.

4. Route each browser to its own cable, in Windows:
   **Settings → System → Sound → Volume mixer**, find each browser in the app
   list, and set its **Output** to **CABLE-A Input** (browser 1) and
   **CABLE-B Input** (browser 2).
   - You won't hear them anymore (they're going to the virtual cables). To still
     hear one, enable "Listen to this device" on that cable's recording device,
     or use VoiceMeeter to monitor.

5. In the transcriber: **Add custom stream** → Source = **pc audio** → pick
   **"CABLE-A Output"** (the recording side), name it e.g. "Fubo A", give it a
   color. Repeat for **"CABLE-B Output"** as "Fubo B".

Both now transcribe as independent sectors, side by side, just like two scanner
feeds.

## Simpler alternative

If you only ever watch **one** thing at a time, skip all of this — just use the
built-in **PC audio / Stereo Mix** source and switch what's playing.
