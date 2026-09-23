# SivletLabs Official Website & Portal

> **System-1 Decision Models & Evaluation Infrastructure**  
> Typed decisions. Calibrated probabilities. Pay-per-call via x402 on Solana Mainnet with Meteora DLMM. Revenue buys back and burns the token.

This repository contains the production-ready, standalone official brand homepage and portal for **SivletLabs**. It is engineered with a cyber-minimalist, high-tech dark aesthetic, featuring real-time cluster telemetry, an interactive x402 API test workbench, and a detailed visualization of the deflationary token buyback loop.

---

## 🌟 Highlights & Features

- **Zero-Build Architecture**: Powered by Tailwind CSS CDN, Lucide Icons, and `@solana/web3.js`. Pure standalone HTML5/CSS3/ES6 with no Node.js compilation, Webpack, or Vite bundling required.
- **Cyber-Minimalist Dark Aesthetic**: Deep obsidian backgrounds (`#07080b`), subtle grid overlays, glassmorphic blur panels, and animated glowing borders.
- **Solana Web3 & Meteora DLMM Native**:
  - Universal Solana wallet connection (Phantom, Solflare, Backpack).
  - Ed25519 Session Key vouchers: 1-time signature authorizes 24 hours of continuous inference with **zero recurring popups** and 12ms edge latency.
  - Direct links to Meteora DLMM pool (`https://app.meteora.ag/`) and Solscan explorer (`https://solscan.io`).
  - Dynamic volatility fee visualization and Alpha Vault fair launch anti-sniper protection.
- **Dual Foundation Pillars**: Highlights **Sivlet-Decision-1** (non-autoregressive sub-15ms inference engine) and **jev-eval** (Gymnasium-compliant RL & benchmarking zoo with strictly proper scoring rules).
- **Core Economic Flywheel ("How Money Flows")**: 5-step visual pipeline depicting how Agent API calls generate USDC revenue via x402 on Solana, route to the Treasury PDA, execute automated Jupiter TWAP buybacks on Meteora DLMM, and burn $SIVLET to the Solana burn sink (`11111111111111111111111111111111`).
- **Live Cluster Telemetry Dashboard**: Real-time ticker tracking API calls, USDC protocol revenue, burned $SIVLET, P50 latency (12ms), Maze 8x8 win rate (100%), and strict Brier calibration (-0.042).
- **Interactive Developer Sandbox**:
  - Switchable code snippets: `cURL (x402 Solana)`, `Python (httpx + solana-py)`, `x402 Agent Tool (TypeScript)`.
  - One-click copy-to-clipboard with visual feedback.
  - Interactive "Live Test Simulator" allowing visitors to test Prompt Injection Guardrails, Dispatch Routing, and Context Compaction in ~12ms.

---

## 🚀 Local Preview & Development

You can preview the website locally using any of the following methods:

### Option 1: Direct File Open (Immediate)
Simply double-click `index.html` or run in macOS terminal:
```bash
open index.html
```

### Option 2: Using Node `npx serve`
```bash
npx serve . -l 3000
```
Then visit: [http://localhost:3000](http://localhost:3000)

### Option 3: Using Python Built-in Server
```bash
python3 -m http.server 8080
```
Then visit: [http://localhost:8080](http://localhost:8080)

---

## 🌐 Production Deployment

Because `index.html` is completely self-contained with no server-side build steps, deployment to global edge CDNs takes under 30 seconds:

### Deploy to GitHub Pages
1. Push changes to branch `main`.
2. The GitHub Pages deployment action automatically publishes to `https://sivletlabs.github.io/`.

---

## 📁 File Structure

```
website/
├── assets/          # Brand assets and logos
├── index.html       # Complete, standalone, responsive homepage application
└── README.md        # Deployment guide, architecture overview, and instructions
```

---

## 📄 License & Attribution

Distributed under the MIT License. Developed by **SivletLabs** (2026).
- Official Repository: [https://github.com/SivletLabs/sivletlabs.github.io](https://github.com/SivletLabs/sivletlabs.github.io)
- Evaluation Harness: `jev-eval` v0.2.0
- Model Engine: `Sivlet-Decision-1`
- Network: Solana Mainnet · Meteora DLMM
