# Is this possible in Webflow?
Yes, absolutely. You can achieve the exact same behavior in Webflow by following these steps:

---

## Step A: Add KaTeX to your Webflow Site
Go to your Page Settings (or Project Settings) and paste this into the Inside tag section:

<!-- KaTeX CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">

<!-- KaTeX JS -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>

---

## Step B: Add the Rendering Script
Paste this into the Before tag section. This is a simplified version of your project's script that will work in any browser:

<script>
document.addEventListener("DOMContentLoaded", function() {
  // Find all <k-math> tags
  const mathElements = document.querySelectorAll('k-math');
  
  mathElements.forEach(el => {
    const latex = el.textContent.trim();
    const isBlock = el.hasAttribute('block');

    try {
      // Use the KaTeX library to render the math
      window.katex.render(latex, el, {
        displayMode: isBlock,
        throwOnError: false
      });
      // Match the styling from your current project
      el.style.visibility = 'visible';
      el.style.display = isBlock ? 'block' : 'inline-block';
    } catch (err) {
      console.error("Math render error:", err);
    }
  });
});
</script>


---
## Your Webflow Embed Links:
You can now use these links in your Webflow iframes:

- DBO Figure: https://harelpanker.github.io/impala_blog/embed/dbo-figure/
- DeepEP Figure: https://harelpanker.github.io/impala_blog/embed/deepep-figure/
- EPLB Figure: https://harelpanker.github.io/impala_blog/embed/eplb-figure/