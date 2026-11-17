# Mole map — capture

Phone web app for the standard-pose capture protocol: pose guide, live body keypoints, shutter
gating, tap-to-mark moles, and on-device placement of each mole on a 3D body template (the
fitter and ray cast run in the browser; no photo or data leaves the phone).

Open on a phone: https://izayour.github.io/molemap-capture/

Files: `index.html` (the app), `molemap.js` (fit + ray cast), `molemap_template.bin` (Anny
body template, 1.2 MB), `molemap_protocol.json` (the 13-shot protocol). Needs internet on first
open for MediaPipe (jsDelivr + Google model host); cached afterwards.
