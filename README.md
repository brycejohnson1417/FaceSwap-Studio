# FaceSwap Studio

FaceSwap Studio is a creative AI prototype for identity-aware visual transformation. It explores how reference images, pose/style analysis, and generated output controls can be combined into a more deliberate creative workflow than a single prompt box.

View the AI Studio prototype: https://ai.studio/apps/c8552a40-e516-4b7c-bc87-036e199c0d2f

## What It Explores

- Reference-image driven creative generation.
- Face and pose analysis using MediaPipe vision tooling.
- A guided workflow for separating identity reference, style direction, and final output review.
- Consent-sensitive product boundaries around image generation.

## Technical Notes

- React and Vite frontend.
- Gemini API integration through `@google/genai`.
- MediaPipe vision tasks for local image/face analysis.
- Motion and lucide-react for interaction and interface details.

## Current Status

This is a prototype source repo. It is useful for exploring the product workflow and interaction model, but it is not a hardened production image-processing service.

Before using this as a public standalone app, move model calls behind a server-side API route, add rate limits, define upload-retention behavior, and add explicit acceptable-use language for identity and likeness workflows.

## Run Locally

Prerequisite: Node.js.

1. Install dependencies:
   `npm install`
2. Create a local environment file based on `.env.example`.
3. Add your own Gemini API key locally.
4. Run the app:
   `npm run dev`

## API Key Boundary

Do not deploy this Vite app with a private Gemini key embedded into browser JavaScript. If deploying outside AI Studio, use a server-side API route or an explicit visitor-provided key flow.

## AI-Assisted Build Note

This prototype was built with AI assistance. The important engineering work is the product framing, workflow structure, safety boundary identification, iteration, and understanding where the prototype stops short of production readiness.

## Related Public Notes

See the combined prototype overview repo: https://github.com/brycejohnson1417/ai-studio-prototype-overviews
