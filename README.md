# IFDL Image Forensics

Chrome extension that scans images in your social feed and checks them for manipulation via an IFDL (Image Forgery Detection & Localization) model served behind a REST API.

## Structure

- `extension/` — Chrome extension (Manifest V3). Finds feed images, sends them to the API, badges flagged results.
- `api/` — REST API running TruFor + HiFi-IFDL, the two models that flag and localize manipulation.

## Extension setup

1. Go to `chrome://extensions`
2. Enable Developer mode
3. Load unpacked → select the `extension` folder
4. Set the API endpoint in the extension popup (defaults to `http://localhost:8000/analyze`)

## API setup

See [`api/README.md`](api/README.md) for full setup (Docker Compose, one manual
weights download, and how to test it).
