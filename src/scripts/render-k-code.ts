// Use highlight.js from CDN (loaded in BlogPost.astro head)
// We'll access it via window.hljs

export {}; // Make this a module

// Type definition for window.hljs
declare global {
  interface Window {
    hljs?: {
      highlight: (code: string, options: { language: string; ignoreIllegals?: boolean }) => { value: string; relevance: number; language: string };
      highlightAuto: (code: string) => { value: string; relevance: number; language: string };
      listLanguages: () => string[];
    };
    __impalaCodeRAF?: number;
  }
}

function renderCodeEl(el: HTMLElement) {
  // If something already rendered this element, don't touch it.
  if (el.dataset.rendered) return;

  // Get raw text content. For blocks, we might want to trim only leading/trailing newlines
  // but preserve indentation.
  let code = el.textContent ?? '';
  
  // Simple trim for start/end, but be careful with indentation
  // If it's a block, we usually want to dedent common indentation
  const isBlock = el.hasAttribute('block');
  
  if (isBlock) {
    // Remove leading newline if present
    if (code.startsWith('\n')) code = code.substring(1);
    // Remove trailing whitespace
    code = code.trimEnd();
  } else {
    code = code.trim();
  }

  if (!code) return;

  const lang = el.getAttribute('lang') || 'plaintext';

  // Clear content
  el.textContent = '';

  // Create wrapper
  // We use a pre tag for blocks to preserve formatting, span for inline
  const wrapper = document.createElement(isBlock ? 'pre' : 'span');
  wrapper.className = isBlock ? 'k-code-block nohighlight' : 'k-code-inline';
  // Prevent Prism.js from processing
  wrapper.setAttribute('data-prismjs-ignore', 'true');

  // Create code element
  const codeEl = document.createElement('code');
  // Add markers to prevent Prism.js (Distill template) from re-processing
  if (lang !== 'plaintext') {
    codeEl.className = `language-${lang}`;
  }
  // These attributes/classes tell Prism.js to skip this element
  codeEl.classList.add('nohighlight');
  codeEl.setAttribute('data-prismjs-ignore', 'true');
  
  codeEl.textContent = code;

  wrapper.appendChild(codeEl);

  // Add copy button for blocks
  if (isBlock) {
    const header = document.createElement('div');
    header.className = 'k-code-header';
    
    // Optional: Add language label
    if (lang && lang !== 'plaintext') {
      const langLabel = document.createElement('span');
      langLabel.className = 'k-code-lang';
      langLabel.textContent = lang;
      header.appendChild(langLabel);
    }

    const copyBtn = document.createElement('button');
    copyBtn.className = 'k-code-copy-btn';
    copyBtn.title = 'Copy code';
    copyBtn.innerHTML = `
      <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <svg class="check-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(code);
      copyBtn.classList.add('copied');
      const copyIcon = copyBtn.querySelector('.copy-icon') as HTMLElement;
      const checkIcon = copyBtn.querySelector('.check-icon') as HTMLElement;
      
      if (copyIcon && checkIcon) {
        copyIcon.style.display = 'none';
        checkIcon.style.display = 'block';
      }

      setTimeout(() => {
        copyBtn.classList.remove('copied');
        if (copyIcon && checkIcon) {
          copyIcon.style.display = 'block';
          checkIcon.style.display = 'none';
        }
      }, 2000);
    };
    
    // If we have a header content (lang or button), wrap code
    // Actually, usually the button floats or is in a header bar.
    // Let's put the button inside the wrapper, absolutely positioned or flex.
    // But wrapper is 'pre', so it's tricky.
    // Let's wrap the 'pre' in a 'div' relative container if it's a block.
    
    // Re-structure for block:
    // <div class="k-code-container">
    //   <div class="k-code-actions">...</div>
    //   <pre><code>...</code></pre>
    // </div>
    
    // But 'wrapper' is currently the 'pre'.
    // Let's change wrapper to be the container div for blocks.
  }

  // Highlight using window.hljs from CDN - wait for it if needed
  const highlight = () => {
    if (lang === 'plaintext') return;
    
    const doHighlight = () => {
      const hljs = window.hljs;
      if (!hljs) {
        setTimeout(doHighlight, 50);
        return;
      }
      
      try {
        const result = hljs.highlight(code, { language: lang, ignoreIllegals: true });
        const applyHighlight = () => {
          codeEl.innerHTML = result.value;
          codeEl.classList.add('hljs');
        };
        
        applyHighlight();
        
        // Re-apply if Distill/Prism overwrites our highlighting
        const observer = new MutationObserver(() => {
          // Check if our hljs spans were removed
          if (!codeEl.innerHTML.includes('hljs-')) {
            observer.disconnect();
            applyHighlight();
            // Re-observe after re-applying
            setTimeout(() => {
              observer.observe(codeEl, { childList: true, subtree: true, characterData: true });
            }, 100);
          }
        });
        
        observer.observe(codeEl, { childList: true, subtree: true, characterData: true });
      } catch {
        // Fallback to auto-detection
        try {
          const hljs = window.hljs;
          if (hljs) {
            const result = hljs.highlightAuto(code);
            codeEl.innerHTML = result.value;
            codeEl.classList.add('hljs');
          }
        } catch {
          // Silent fail - code will display without highlighting
        }
      }
    };
    
    doHighlight();
  };

  if (isBlock) {
    const container = document.createElement('div');
    container.className = 'k-code-container';
    
    const actions = document.createElement('div');
    actions.className = 'k-code-actions';
    
    if (lang && lang !== 'plaintext') {
        const langLabel = document.createElement('span');
        langLabel.className = 'k-code-lang';
        langLabel.textContent = lang;
        actions.appendChild(langLabel);
    }

    const copyBtn = document.createElement('button');
    copyBtn.className = 'k-code-copy-btn';
    copyBtn.ariaLabel = 'Copy code';
    copyBtn.innerHTML = `
      <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <svg class="check-icon" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(code);
      copyBtn.classList.add('copied');
      const copyIcon = copyBtn.querySelector('.copy-icon') as HTMLElement;
      const checkIcon = copyBtn.querySelector('.check-icon') as HTMLElement;
      
      if (copyIcon && checkIcon) {
        copyIcon.style.display = 'none';
        checkIcon.style.display = 'block';
      }

      setTimeout(() => {
        copyBtn.classList.remove('copied');
        if (copyIcon && checkIcon) {
          copyIcon.style.display = 'block';
          checkIcon.style.display = 'none';
        }
      }, 2000);
    };

    actions.appendChild(copyBtn);
    container.appendChild(actions);
    container.appendChild(wrapper); // wrapper is the pre tag
    el.appendChild(container);
  } else {
    el.appendChild(wrapper);
  }

  el.dataset.rendered = 'true';
  el.style.visibility = 'visible';
  el.style.display = isBlock ? 'block' : 'inline-block';

  // Call highlight AFTER the element is attached to the DOM
  highlight();
}

function renderAllCode(root: ParentNode) {
  root.querySelectorAll('k-code').forEach((node) => {
    if (node instanceof HTMLElement) renderCodeEl(node);
  });
}

function scheduleRenderAll() {
  if (typeof window === 'undefined') return;
  const w = window;
  if (w.__impalaCodeRAF) return;
  w.__impalaCodeRAF = window.requestAnimationFrame(() => {
    w.__impalaCodeRAF = undefined;
    renderAllCode(document);
  });
}

if (typeof window !== 'undefined') {
  const start = () => renderAllCode(document);
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }

  // Also listen for hljs load if it's async
  // But usually DOMContentLoaded is enough if script is in head or end of body.
  
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        const tag = node.tagName.toLowerCase();
        if (tag === 'k-code' || node.querySelector('k-code')) {
          scheduleRenderAll();
          return;
        }
      }
    }
  });

  observer.observe(document.documentElement, { childList: true, subtree: true });
}
