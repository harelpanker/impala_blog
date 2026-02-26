# Impala AI Technical Blog

An [Astro](https://astro.build/) blog with interactive visualizations built with MDX, D3.js, and KaTeX.

## Quick Start

### Prerequisites

- **[Bun](https://bun.sh/)** (v1.x) — JavaScript runtime and package manager.
  Install: `curl -fsSL https://bun.sh/install | bash`
- **Node.js 20+** (Bun handles most things, but some tooling may need Node)

### Install & Run

```bash
# Install dependencies
bun install

# Start the dev server (http://localhost:4321)
bun run dev

# Build static output to dist/
bun run build

# Preview the production build locally
bun run preview
```

## Project Structure

```
├── public/                     # Static assets copied as-is to dist/
│   ├── favicon.svg
│   └── scripts/
│       └── moe-widget.js      # Interactive MoE visualization widget
├── src/
│   ├── content/
│   │   ├── config.ts           # Content collection schema
│   │   └── blog/               # Blog posts (MDX files)
│   │       ├── wide_ep_moe_dbo_deepep_nvl72.mdx
│   │       ├── wide-ep-part-1-wire-model.mdx
│   │       ├── wide-ep-part-2-dbo-kernels-hardware.mdx
│   │       └── wide-ep-part-3-balancing-portability.mdx
│   ├── components/             # Astro components (interactive figures, nav, etc.)
│   │   ├── CommRoofline.astro
│   │   ├── DBOInteractiveFigure.astro
│   │   ├── DBOTimeline.astro
│   │   ├── DeepEPAnalysis.astro
│   │   ├── DeepEPFigure.astro
│   │   ├── DeepEPMechanics.astro
│   │   ├── EPLBFigure.astro
│   │   └── SeriesNav.astro
│   ├── layouts/
│   │   └── BlogPost.astro      # Base layout for all blog posts
│   ├── pages/
│   │   ├── index.astro         # Blog index / homepage
│   │   └── blog/[slug].astro   # Dynamic route for individual posts
│   └── scripts/
│       ├── render-k-code.ts    # KaTeX rendering helpers
│       └── render-d-math.ts    # Math rendering helpers
├── dist/                       # Build output (static HTML/CSS/JS) — git-ignored
├── astro.config.mjs            # Astro configuration
├── tsconfig.json
├── package.json
└── bun.lock
```

## Build Output & Webflow Integration

Running `bun run build` generates a fully static site in the `dist/` folder. Each blog post becomes its own `index.html` at:

```
dist/
├── index.html                              # Homepage
├── blog/
│   ├── wide_ep_moe_dbo_deepep_nvl72/index.html
│   ├── wide-ep-part-1-wire-model/index.html
│   ├── wide-ep-part-2-dbo-kernels-hardware/index.html
│   └── wide-ep-part-3-balancing-portability/index.html
├── _astro/                                 # Hashed CSS & JS bundles
└── scripts/
    └── moe-widget.js
```

### How to use with Webflow

The build output is **plain static HTML + CSS + JS** — no server required. To integrate with Webflow:

1. **Embed approach (recommended):** Use Webflow's [Custom Code Embed](https://university.webflow.com/lesson/custom-code-embed) or page-level custom code to embed the blog post HTML/CSS/JS directly into a Webflow page.
2. **Subdomain / subdirectory approach:** Host the `dist/` folder on its own subdomain (e.g., `blog.impala.ai`) using any static host (Vercel, Netlify, Cloudflare Pages, AWS S3 + CloudFront) and link to it from Webflow.
3. **Reverse proxy approach:** Use Cloudflare Workers or similar to serve `blog.impala.ai/blog/*` from the static build while keeping the rest on Webflow.

### Important notes for the Webflow developer

- **Interactive components**: Many blog posts contain interactive D3.js visualizations and KaTeX math. These require the JS/CSS bundles in `_astro/` to work. Make sure all assets are served together.
- **Self-contained pages**: Each page in `dist/` is fully self-contained with inlined critical CSS and script references relative to the site root.
- **No API / backend**: This is a pure static site. No server, database, or API is needed.
- **Fonts & styles**: The blog uses its own styles (scoped via Astro). Review `_astro/*.css` for any conflicts with Webflow's global styles.

## Adding a New Blog Post

1. Create a new `.mdx` file in `src/content/blog/`
2. Add frontmatter (title, date, description, etc.) matching the schema in `src/content/config.ts`
3. Write content using Markdown + any Astro components from `src/components/`
4. Run `bun run build` to generate the static output

## Technologies

- **[Astro](https://astro.build/)** — Static site generator
- **[MDX](https://mdxjs.com/)** — Markdown with JSX components
- **[D3.js](https://d3js.org/)** — Data visualizations
- **[KaTeX](https://katex.org/)** — Math typesetting
- **[Bun](https://bun.sh/)** — JavaScript runtime and package manager

## License

(c) 2026 Impala AI. All rights reserved.
