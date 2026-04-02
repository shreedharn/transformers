/**
 * Hybrid MathML rendering stabilizer for Chrome/Edge/Safari.
 * Fixes the "broken on first load, fine on refresh" issue by:
 *  1. Waiting for font loading (Font Loading API)
 *  2. Forcing MathML reflow after fonts are ready
 *  3. Re-running after a short delay in case of lazy font swaps
 */
(function() {
  function fixMathMLRendering() {
    const mathNodes = document.querySelectorAll("math");
    if (!mathNodes.length) return;

    mathNodes.forEach(el => {
      const h = el.offsetHeight; // trigger layout read
      el.style.display = "none";
      void el.offsetHeight;       // invalidate render tree
      el.style.display = "";
    });
  }

  function ready(fn) {
    if (document.readyState === "complete" || document.readyState === "interactive") {
      fn();
    } else {
      document.addEventListener("DOMContentLoaded", fn);
    }
  }

  ready(() => {
    // Wait for all fonts to finish loading
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        fixMathMLRendering();
        setTimeout(fixMathMLRendering, 300);
      });
    } else {
      // Fallback for older browsers
      window.addEventListener("load", fixMathMLRendering);
    }
  });
})();
