# SivletLabs Official Website & Portal

> **System-1 Decision Models & Evaluation Infrastructure**  
> Typed decisions. Calibrated probabilities. Pay-per-call via x402. Revenue buys back and burns the token.

This repository contains the production-ready, standalone official brand homepage and portal for **SivletLabs**. It is engineered with a cyber-minimalist, high-tech dark aesthetic, featuring real-time cluster telemetry, an interactive x402 API test workbench, and a detailed visualization of the deflationary token buyback loop.

---

## 🌟 Highlights & Features

- **Zero-Build Architecture**: Powered by Tailwind CSS CDN and Lucide Icons. Pure standalone HTML5/CSS3/ES6 with no Node.js compilation, Webpack, or Vite bundling required.
- **Cyber-Minimalist Dark Aesthetic**: Deep obsidian backgrounds (`#05070d`), subtle cyan grid overlays, glassmorphic blur panels (`backdrop-blur-xl`), and animated glowing borders.
- **Dual Foundation Pillars**: Highlights **Sivlet-Jev / Decision-1** (non-autoregressive sub-15ms inference engine) and **jev-eval** (Gymnasium-compliant RL & benchmarking zoo with strictly proper scoring rules).
- **Core Economic Flywheel ("How Money Flows")**: 5-step visual pipeline depicting how Agent API calls generate USDC revenue via x402, route to Treasury, execute automated TWAP buybacks, and burn $SIVLET to `0x0...dead`.
- **Live Cluster Telemetry Dashboard**: Real-time ticker tracking API calls, USDC protocol revenue, burned $SIVLET, P50 latency (12ms), Maze 8x8 win rate (100%), and strict Brier calibration (-0.042).
- **Interactive Developer Sandbox**:
  - Switchable code snippets: `cURL (x402)`, `Python (jev_env / Gymnasium)`, `x402 Agent Tool (TypeScript)`.
  - One-click copy-to-clipboard with visual feedback.
  - Interactive "Live Test Simulator" allowing visitors to test Prompt Injection Guardrails, Dispatch Routing, and Context Compaction in ~12ms.
- **Built-in Litepaper Modal**: Pop-up technical overview detailing the MDP formulation, Strictly Proper Scoring rules, and deflationary economics.

---

## 🚀 Local Preview & Development

You can preview the website locally using any of the following methods:

### Option 1: Direct File Open (Immediate)
Simply double-click `index.html` or run in macOS terminal:
```bash
open /Users/echo/project/SivletLabs/website/index.html
```

### Option 2: Using Node `npx serve`
```bash
cd /Users/echo/project/SivletLabs/website
npx serve . -l 3000
```
Then visit: [http://localhost:3000](http://localhost:3000)

### Option 3: Using Python Built-in Server
```bash
cd /Users/echo/project/SivletLabs/website
python3 -m http.server 8080
```
Then visit: [http://localhost:8080](http://localhost:8080)

### Option 4: Using Bun
```bash
cd /Users/echo/project/SivletLabs/website
bun x serve -p 3000 .
```

---

## 🌐 Production Deployment

Because `index.html` is completely self-contained with no server-side build steps, deployment to global edge CDNs takes under 30 seconds:

### Deploy to Vercel
```bash
cd /Users/echo/project/SivletLabs/website
npx vercel --prod
```

Or connect the GitHub repository directly to Vercel:
- **Framework Preset**: `Other`
- **Root Directory**: `website`
- **Build Command**: *(leave empty)*
- **Output Directory**: `.`

### Deploy to Cloudflare Pages
```bash
cd /Users/echo/project/SivletLabs/website
npx wrangler pages deploy . --project-name=sivletlabs-portal
```

### Deploy to GitHub Pages
1. In your GitHub repository settings, navigate to **Pages**.
2. Under **Build and deployment > Source**, select **Deploy from a branch**.
3. Select branch `main` and folder `/website` (or set up a GitHub Action to deploy `./website` to `gh-pages`).

### Deploy via Docker / Nginx
Create a minimal `Dockerfile`:
```dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 📁 File Structure

```
website/
├── index.html       # Complete, standalone, responsive homepage application
└── README.md        # Deployment guide, architecture overview, and instructions
```

---

## 🛠️ Configuration & Customization

- **Smart Contract Addresses**: Modify lines 390-415 in `index.html` to update the Base L2 token contract, Treasury multi-sig, or TWAP keeper address.
- **API Endpoint URLs**: Modify lines 530-580 to update live API endpoints (`https://api.sivlet.org/v1/systemone`).
- **Telemetry Starting Numbers**: Update `initLiveCounterTicker()` in the `<script>` tag to align with on-chain metrics.

---

## 📄 License & Attribution

Distributed under the MIT License. Developed by **SivletLabs** (2026).
- Official Repository: [https://github.com/SivletLabs/jev-eval](https://github.com/SivletLabs/jev-eval)
- Evaluation Harness: `jev-eval` v0.2.0
- Model Engine: `Sivlet-Jev`
