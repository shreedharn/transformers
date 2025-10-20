(function () {
  // Toggle to true only when you need to investigate; it’s quiet by default.
  const DEBUG = false;
  const log = (...a) => DEBUG && console.log("[MJX]", ...a);
  const warn = (...a) => DEBUG && console.warn("[MJX]", ...a);

  // Guard state to prevent overlapping renders.
  let rendering = false;
  let scheduled = false;
  let lastNavRenderAt = 0;

  // Cheap test to decide whether a subtree might contain LaTeX.
  function maybeHasMath(node) {
    if (!node) return false;
    const t = node.textContent || "";
    return /\$\$|\\\(|\\\[|\\begin\{/.test(t);
  }

  // Debounced render scheduler: coalesces calls into one typeset pass.
  function scheduleTypeset(target, dueMs = 180) {
    if (scheduled) return;
    scheduled = true;
    setTimeout(() => performTypeset(target), dueMs);
  }

  function waitForMathJaxReady(cb) {
    if (window.MathJax && window.MathJax.startup) return cb();
    setTimeout(() => waitForMathJaxReady(cb), 120);
  }

  // Use a brief “quiet window” to let MkDocs finish inserting/reflowing DOM.
  function whenDomQuiet(root, cb, quietMs = 200, ceilingMs = 1200) {
    let last = Date.now();
    const obs = new MutationObserver(() => (last = Date.now()));
    obs.observe(root, { childList: true, subtree: true });

    const start = Date.now();
    (function tick() {
      if (Date.now() - last >= quietMs || Date.now() - start > ceilingMs) {
        obs.disconnect();
        cb();
      } else {
        setTimeout(tick, 80);
      }
    })();
  }

  function performTypeset(target) {
    scheduled = false;
    if (rendering) {
      // If a render is in-flight, try again shortly (but don’t spam).
      warn("Render requested while busy; rescheduling…");
      return scheduleTypeset(target, 160);
    }
    rendering = true;

    waitForMathJaxReady(() => {
      // Give layout a breath and avoid racing MkDocs animations
      whenDomQuiet(target, () => {
        log("Typeset begin");
        window.MathJax.startup.promise
          .then(() => {
            // IMPORTANT: do not call typesetClear on every turn; only when navigating.
            // It can blow away containers mid-flight. We clear only after nav (see below).
            return window.MathJax.typesetPromise([target]);
          })
          .then(() => {
            log("Typeset complete");
          })
          .catch((e) => {
            warn("Typeset error", e);
          })
          .finally(() => {
            rendering = false;
          });
      });
    });
  }

  // One-shot safety retry used only after a navigation render, if needed.
  function safetyRetry(root) {
    // If MathJax missed something because of late content, do a final quick pass.
    scheduleTypeset(root, 220);
  }

  // MkDocs instant navigation hook (Material’s document$ stream).
  function onPageChange(fn) {
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(({ body }) => fn(body, true));
    } else {
      window.addEventListener("load", () => fn(document.body, true), { once: true });
    }
  }

  // Observe late inserts inside the page (tabs, details, search, etc.).
  function observeLateMath(root) {
    const obs = new MutationObserver((muts) => {
      for (const m of muts) {
        for (const n of m.addedNodes) {
          if (n.nodeType === 1 && maybeHasMath(n)) {
            log("Late math detected -> scheduleTypeset");
            scheduleTypeset(root, 180);
            return;
          }
        }
      }
    });
    obs.observe(root, { childList: true, subtree: true });
  }

  // Main hook: runs after every nav and once on initial load.
  onPageChange((root, isNav) => {
    // Clear MathJax’s internal state only on navigation to avoid partial wipes mid-page.
    if (isNav && window.MathJax?.startup?.document) {
      try { window.MathJax.startup.document.clear(); } catch (_) {}
    }

    // Render after MkDocs swaps the DOM; small delay helps Safari + transitions.
    lastNavRenderAt = Date.now();
    scheduleTypeset(root, 240);     // primary pass
    setTimeout(() => safetyRetry(root), 700); // single cleanup pass

    // Watch for late inserts in this page.
    observeLateMath(root);
  });
})();
