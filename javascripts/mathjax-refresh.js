(function () {
  console.log("[MathJax Hybrid+Sync] Initialized");

  // --- detect unrendered LaTeX text ------------------------
  function hasUnrenderedMath(el) {
    if (!el) return false;
    const text = el.textContent || "";
    const rawMath = /\$\$|\\\(|\\\)|\\\[|\\\]|\\begin\{/.test(text);
    const rendered = el.querySelector("mjx-container") !== null;
    const result = rawMath && !rendered;
    console.log(`[MathJax Hybrid+Sync] hasUnrenderedMath → ${result}`);
    return result;
  }

  // --- perform safe rendering with retries -----------------
  function safeTypeset(target, attempt = 1) {
    console.log(`[MathJax Hybrid+Sync] Typeset attempt ${attempt}`);

    if (!window.MathJax || !window.MathJax.startup) {
      console.warn("[MathJax Hybrid+Sync] MathJax not ready — waiting...");
      return setTimeout(() => safeTypeset(target, attempt), 300);
    }

    // Ensure MathJax startup is done before any typeset call
    window.MathJax.startup.promise
      .then(() => {
        console.log("[MathJax Hybrid+Sync] Startup complete, running typeset...");
        window.MathJax.typesetClear();
        return window.MathJax.typesetPromise([target]);
      })
      .then(() => {
        // queue a delayed final render to catch missed nodes
        if (hasUnrenderedMath(target) && attempt < 3) {
          console.warn(
            `[MathJax Hybrid+Sync] Unrendered math detected — retrying (${attempt}/3)...`
          );
          return setTimeout(() => safeTypeset(target, attempt + 1), 400);
        }
        console.log("[MathJax Hybrid+Sync] MathJax rendering complete ✅");
      })
      .catch((e) => {
        console.error("[MathJax Hybrid+Sync] MathJax typeset error:", e);
      });
  }

  // --- re-typeset after MkDocs instant navigation ----------
  function onPageChange(fn) {
    if (window.document$ && window.document$.subscribe) {
      console.log("[MathJax Hybrid+Sync] Subscribed to document$ navigation events");
      window.document$.subscribe(({ body }) => {
        console.log("[MathJax Hybrid+Sync] New page loaded");
        setTimeout(() => fn(body), 200);
      });
    } else {
      window.addEventListener("load", () => fn(document.body), { once: true });
    }
  }

  // Run on every MkDocs page change
  onPageChange((root) => safeTypeset(root, 1));

  // Run once on initial load after MathJax finishes bootstrapping
  setTimeout(() => safeTypeset(document.body, 1), 300);
})();
