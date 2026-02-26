import puppeteer from "puppeteer";
import { mkdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join } from "node:path";

const PORT = 4321;
const BASE = `http://localhost:${PORT}`;
const OUT_DIR = join(import.meta.dir, "pdfs");

const pages = [
  { path: "/blog/wide-ep-part-1-wire-model/", name: "wide-ep-part-1-wire-model" },
  { path: "/blog/wide-ep-part-2-dbo-kernels-hardware/", name: "wide-ep-part-2-dbo-kernels-hardware" },
  { path: "/blog/wide-ep-part-3-balancing-portability/", name: "wide-ep-part-3-balancing-portability" },
  { path: "/blog/wide_ep_moe_dbo_deepep_nvl72/", name: "wide-ep-reference-edition" },
];

async function waitForServer(url: string, maxRetries = 30) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`Server at ${url} did not start in time`);
}

async function main() {
  console.log("Building site...");
  const build = Bun.spawnSync(["bun", "run", "build"], {
    cwd: import.meta.dir,
    stdio: ["inherit", "inherit", "inherit"],
  });
  if (build.exitCode !== 0) {
    throw new Error("Build failed");
  }

  await mkdir(OUT_DIR, { recursive: true });

  console.log("Starting preview server...");
  const preview = spawn("bun", ["run", "preview", "--", "--port", String(PORT)], {
    cwd: import.meta.dir,
    stdio: "pipe",
  });

  try {
    await waitForServer(BASE);
    console.log("Server is up.\n");

    const browser = await puppeteer.launch({ headless: true });

    for (const page of pages) {
      const url = `${BASE}${page.path}`;
      console.log(`Rendering ${page.name}...`);

      const tab = await browser.newPage();
      await tab.goto(url, { waitUntil: "networkidle2", timeout: 60_000 });

      await new Promise((r) => setTimeout(r, 3000));

      const outPath = join(OUT_DIR, `${page.name}.pdf`);
      await tab.pdf({
        path: outPath,
        format: "A4",
        printBackground: true,
        margin: { top: "16mm", bottom: "16mm", left: "12mm", right: "12mm" },
      });

      await tab.close();
      console.log(`  -> saved: pdfs/${page.name}.pdf`);
    }

    await browser.close();
    console.log(`\nDone! All PDFs saved to: ${OUT_DIR}`);
  } finally {
    preview.kill();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
