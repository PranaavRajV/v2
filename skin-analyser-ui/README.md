# Smart Mirror — Skin Analyser UI

A reference-led React/Vite frontend for the skin analyser screen.

## What is included

- Full-screen mirror-style interface inspired by the supplied reference.
- Warm ivory typography, rose-pink accents, champagne and mint highlights.
- Smoky glassmorphism score/profile and recommendation panels.
- Live browser camera preview using `getUserMedia`.
- Facial callout overlay positions and particle texture as a **visual prototype**.
- Responsive layout and a bottom navigation bar.
- Personalized-routine demo popover.

## Important: demo values

All scores, labels, profile values and recommendations in `src/App.jsx` are sample UI data. The current code does **not** run DermaCLIP, perform skin diagnosis, detect facial landmarks, or calculate a clinically meaningful skin-health score. Replace the `DEMO` object with validated outputs from your backend before presenting results as real analysis.

The face markers are currently positioned using percentage coordinates to reproduce the reference composition. They are not yet anchored to MediaPipe landmarks.

## Run on your Mac

1. Install Node.js (LTS) if it is not already installed.
2. Open Terminal and move into this folder.
3. Install dependencies:

   ```bash
   npm install
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open the local URL printed by Vite, usually `http://localhost:5173`.
6. Click **Enable camera** and allow browser camera access.

Camera access generally requires `localhost` or HTTPS. If you use an external camera, choose it in your browser's site camera settings.

## Build for fullscreen / kiosk testing

```bash
npm run build
npm run preview
```

Open the preview URL in Chrome and use fullscreen mode.

## Project structure

```text
skin-analyser-ui/
├── index.html
├── package.json
├── README.md
└── src/
    ├── main.jsx
    ├── App.jsx
    └── styles.css
```

## Next integration steps

1. Add a FastAPI endpoint that returns the model's skin-analysis results.
2. Map each supported model output to the six visible callouts.
3. Define and validate the score calibration / aggregation logic.
4. Add MediaPipe face landmarks and anchor callouts to tracked facial regions.
5. Tune panel opacity, typography and contrast on the actual rear-projection film and glass.

## Design notes

- Primary UI font stack: Avenir Next / Avenir with Inter fallback.
- Greeting: Allura with cursive fallback.
- The stylesheet currently imports Google Fonts for Inter and Allura. For an offline kiosk, download and self-host licensed font files or remove the import and use local fallbacks.
- Layout and opacity values are initial design approximations; final calibration must happen on the actual mirror.
