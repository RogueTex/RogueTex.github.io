---
layout: default
title: "Blog"
permalink: /blog
---

<div style="max-width: 900px; margin: 0 auto; padding: 2rem 1rem;">
  <h1 style="font-size: 2rem; font-weight: 700; margin-bottom: 1.5rem;">Projects & Notes</h1>
  <p style="color: #4b5563; margin-bottom: 2rem;">
    A quick overview of a few projects I built recently—what they do and how they work under the hood.
  </p>

  <div style="display: grid; grid-template-columns: 1fr; gap: 1.25rem;">

    <!-- TradingBot -->
    <article style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.25rem;">
      <h2 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.5rem 0;">TradingBot — Backtesting Engine</h2>
      <p style="color: #374151; margin: 0 0 0.75rem 0;">
        A lightweight backtesting framework to simulate and evaluate trading strategies on historical market data.
      </p>
      <div style="color: #4b5563; font-size: 0.975rem; line-height: 1.6;">
        <strong>How it’s built:</strong>
        <ul style="margin: 0.5rem 0 0 1.25rem;">
          <li>Data ingestion, cleaning, and resampling for consistent timeframes</li>
          <li>Strategy module with pluggable signals, position sizing, and risk controls</li>
          <li>Backtest loop for order simulation, PnL tracking, and performance metrics</li>
          <li>Summary report with returns, drawdown, and win/loss stats</li>
        </ul>
      </div>
      <p style="margin-top: 0.75rem;">
        <a href="https://github.com/RogueTex/TradingBot" style="display: inline-flex; align-items: center; gap: 0.5rem; color: #111827; text-decoration: none; border: 1px solid #e5e7eb; padding: 0.4rem 0.7rem; border-radius: 8px;">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.58.11.8-.25.8-.56 0-.27-.01-1.16-.02-2.11-3.2.7-3.88-1.36-3.88-1.36-.53-1.35-1.3-1.7-1.3-1.7-1.06-.72.08-.7.08-.7 1.18.08 1.8 1.21 1.8 1.21 1.04 1.79 2.73 1.27 3.4.97.11-.75.41-1.27.75-1.56-2.56-.29-5.26-1.28-5.26-5.69 0-1.26.45-2.29 1.2-3.1-.12-.3-.52-1.51.11-3.15 0 0 .98-.31 3.2 1.18a11.16 11.16 0 0 1 2.92-.39c.99 0 1.99.13 2.93.39 2.21-1.49 3.19-1.18 3.19-1.18.63 1.64.23 2.85.11 3.15.75.81 1.2 1.84 1.2 3.1 0 4.42-2.71 5.39-5.29 5.68.42.36.8 1.08.8 2.18 0 1.58-.02 2.85-.02 3.24 0 .31.21.68.81.56A10.99 10.99 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
          View on GitHub
        </a>
      </p>
    </article>

    <!-- AdvancedInsights -->
    <article style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.25rem;">
      <h2 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.5rem 0;">AdvancedInsights — Mini Perplexity</h2>
      <p style="color: #374151; margin: 0 0 0.75rem 0;">
        A basic Perplexity-style app that fetches live results and produces concise summaries to answer queries quickly.
      </p>
      <div style="color: #4b5563; font-size: 0.975rem; line-height: 1.6;">
        <strong>How it’s built:</strong>
        <ul style="margin: 0.5rem 0 0 1.25rem;">
          <li>Frontend: React + Vite for a fast, responsive UI</li>
          <li>Backend: FastAPI with HTTP scraping and summarization pipeline</li>
          <li>Summarization with HuggingFace transformers for readable outputs</li>
          <li>Deployed with static hosting (frontend) and a lightweight API service</li>
        </ul>
      </div>
      <p style="margin-top: 0.75rem;">
        <a href="https://github.com/RogueTex/AdvancedInsights" style="display: inline-flex; align-items: center; gap: 0.5rem; color: #111827; text-decoration: none; border: 1px solid #e5e7eb; padding: 0.4rem 0.7rem; border-radius: 8px;">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.58.11.8-.25.8-.56 0-.27-.01-1.16-.02-2.11-3.2.7-3.88-1.36-3.88-1.36-.53-1.35-1.3-1.7-1.3-1.7-1.06-.72.08-.7.08-.7 1.18.08 1.8 1.21 1.8 1.21 1.04 1.79 2.73 1.27 3.4.97.11-.75.41-1.27.75-1.56-2.56-.29-5.26-1.28-5.26-5.69 0-1.26.45-2.29 1.2-3.1-.12-.3-.52-1.51.11-3.15 0 0 .98-.31 3.2 1.18a11.16 11.16 0 0 1 2.92-.39c.99 0 1.99.13 2.93.39 2.21-1.49 3.19-1.18 3.19-1.18.63 1.64.23 2.85.11 3.15.75.81 1.2 1.84 1.2 3.1 0 4.42-2.71 5.39-5.29 5.68.42.36.8 1.08.8 2.18 0 1.58-.02 2.85-.02 3.24 0 .31.21.68.81.56A10.99 10.99 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
          View on GitHub
        </a>
      </p>
    </article>

    <!-- ContentGenerator_RooHackathon -->
    <article style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.25rem;">
      <h2 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.5rem 0;">ContentGenerator — Roo Hackathon Finalist</h2>
      <p style="color: #374151; margin: 0 0 0.75rem 0;">
        A content generation tool that creates blogs, newsletters, and ad copy with configurable tones and lengths.
      </p>
      <div style="color: #4b5563; font-size: 0.975rem; line-height: 1.6;">
        <strong>How it’s built:</strong>
        <ul style="margin: 0.5rem 0 0 1.25rem;">
          <li>Frontend: React + Tailwind for a clean, responsive UI</li>
          <li>Backend: Node + Express, integrates with Requesty / Google Gemini</li>
          <li>Markdown rendering and content history for easy reuse</li>
          <li>CI/CD with GitHub Actions and static hosting for the frontend</li>
        </ul>
      </div>
      <p style="margin-top: 0.75rem;">
        <a href="https://github.com/RogueTex/ContentGenerator_RooHackathon" style="display: inline-flex; align-items: center; gap: 0.5rem; color: #111827; text-decoration: none; border: 1px solid #e5e7eb; padding: 0.4rem 0.7rem; border-radius: 8px;">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.58.11.8-.25.8-.56 0-.27-.01-1.16-.02-2.11-3.2.7-3.88-1.36-3.88-1.36-.53-1.35-1.3-1.7-1.3-1.7-1.06-.72.08-.7.08-.7 1.18.08 1.8 1.21 1.8 1.21 1.04 1.79 2.73 1.27 3.4.97.11-.75.41-1.27.75-1.56-2.56-.29-5.26-1.28-5.26-5.69 0-1.26.45-2.29 1.2-3.1-.12-.3-.52-1.51.11-3.15 0 0 .98-.31 3.2 1.18a11.16 11.16 0 0 1 2.92-.39c.99 0 1.99.13 2.93.39 2.21-1.49 3.19-1.18 3.19-1.18.63 1.64.23 2.85.11 3.15.75.81 1.2 1.84 1.2 3.1 0 4.42-2.71 5.39-5.29 5.68.42.36.8 1.08.8 2.18 0 1.58-.02 2.85-.02 3.24 0 .31.21.68.81.56A10.99 10.99 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>
          View on GitHub
        </a>
      </p>
    </article>

  </div>
</div>
