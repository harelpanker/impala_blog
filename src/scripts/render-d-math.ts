import katex from 'katex';
import 'katex/dist/katex.min.css';

function renderMathEl(el: HTMLElement) {
  // If something already rendered this element, don't touch it.
  if (el.querySelector('.katex')) return;

  const latex = (el.textContent ?? '').trim();
  if (!latex) return;

  const displayMode = el.hasAttribute('block');

  try {
    el.textContent = '';
    katex.render(latex, el, {
      displayMode,
      throwOnError: false,
      // Distill-style content can have non-standard macros; be permissive.
      strict: false,
      trust: true,
    });
  } catch {
    // Fallback: at least show the raw string.
    el.textContent = latex;
  } finally {
    // Distill's template can hide math until it's rendered.
    el.style.visibility = 'visible';
    // Also guard against display:none (common "hide until upgraded" pattern).
    el.style.display = displayMode ? 'block' : 'inline-block';
  }
}

function renderAllMath(root: ParentNode) {
  // Prefer our own tag (<k-math>), but keep <d-math> support for legacy content.
  root.querySelectorAll('k-math, d-math').forEach((node) => {
    if (node instanceof HTMLElement) renderMathEl(node);
  });
}

function scheduleRenderAll() {
  // Batch many DOM mutations into one render pass.
  if (typeof window === 'undefined') return;
  const w = window as unknown as { __impalaMathRAF?: number };
  if (w.__impalaMathRAF) return;
  w.__impalaMathRAF = window.requestAnimationFrame(() => {
    w.__impalaMathRAF = undefined;
    renderAllMath(document);
  });
}

if (typeof window !== 'undefined') {
  const start = () => renderAllMath(document);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        const tag = node.tagName.toLowerCase();
        if (tag === 'k-math' || tag === 'd-math' || node.querySelector('k-math, d-math')) {
          scheduleRenderAll();
          return;
        }
      }
    }
  });

  observer.observe(document.documentElement, { childList: true, subtree: true });
}

