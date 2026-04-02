(function () {
  /* --------------------------------------------------------------------------
   * 1️⃣  WARM-UP STYLESHEET
   * --------------------------------------------------------------------------
   *  Creates a guaranteed <style> before MathJax injects any of its own.
   *  Prevents "t.sheet is null" during insertRule() on first paint.
   * ------------------------------------------------------------------------ */
  window.MathJax = window.MathJax || {};
  window.MathJax.startup = window.MathJax.startup || {};
  window.MathJax.startup.ready = () => {
    MathJax.startup.defaultReady();
    try {
      const mjxStyle = document.createElement("style");
      mjxStyle.dataset.mjxStyle = "warmup";
      document.head.appendChild(mjxStyle);
      mjxStyle.sheet.insertRule(".mjx-container{}");
      console.log("[MathJax Debug] ✅ Warm-up stylesheet inserted and live.");
    } catch (e) {
      console.warn("[MathJax Debug] ⚠️ Warm-up stylesheet insert failed:", e);
    }
  };

  /* --------------------------------------------------------------------------
   * 2️⃣  PAGE-CHANGE HANDLER
   * --------------------------------------------------------------------------
   *  Hook into MkDocs Material’s instant-navigation (window.document$) or
   *  fall back to normal window.load.  Ensures re-typesetting on every page.
   * ------------------------------------------------------------------------ */
  const onPageChange = (fn) => {
    if (window.document$ && window.document$.subscribe) {
      console.log("[MathJax Debug] Subscribed to document$ navigation events");
      window.document$.subscribe(({ body }) => {
        console.log("[MathJax Debug] New page loaded via instant navigation");
        fn(body);
      });
    } else {
      console.log("[MathJax Debug] Using fallback on window.load");
      window.addEventListener("load", () => fn(document.body), { once: true });
    }
  };

  /* --------------------------------------------------------------------------
   * 3️⃣  MAIN RE-TYPESET ROUTINE
   * ------------------------------------------------------------------------ */
  onPageChange((root) => {
    let attempts = 0;

    /* ---- 3a. Catch insertRule errors + retry ---- */
    function trapMathJaxStyleErrors() {
      // Catch runtime insertRule() exceptions
      window.addEventListener(
        "error",
        (e) => {
          if (
            e.message?.includes("insert css rule") ||
            e.message?.includes("insertRule")
          ) {
            console.warn(
              "[MathJax Debug] ⚠️ insertRule error caught → scheduling re-render"
            );
            e.preventDefault?.();
            setTimeout(() => window.MathJax?.typesetPromise?.([root]), 250);
          }
        },
        true
      );

      // 2️⃣ Watch for deferred <style> attachments
      const observer = new MutationObserver((muts) => {
        muts.forEach((m) => {
          m.addedNodes.forEach((node) => {
            if (node.nodeName === "STYLE") {
              try {
                void node.sheet; // throws if not yet attached
              } catch {
                console.warn(
                  "[MathJax Debug] Deferred stylesheet not yet ready; will retry typeset..."
                );
                setTimeout(() => {
                  if (window.MathJax?.typesetPromise) {
                    window.MathJax.typesetPromise([root])
                      .then(() =>
                        console.log(
                          "[MathJax Debug] Retypeset after stylesheet attach"
                        )
                      )
                      .catch((e) =>
                        console.error("[MathJax Debug] Retypeset error:", e)
                      );
                  }
                }, 200);
              }
            }
          });
        });
      });
      observer.observe(document.head, { childList: true });
    }

    trapMathJaxStyleErrors(); // ✅ install observers before rendering

    /* ---- 3b. Detect unrendered TeX ---- */
    function hasUnrenderedMath(el) {
      const txt = el.innerText || "";
      const unrendered = /\$\$|\\begin\{/.test(txt);
      console.log(
        `[MathJax Debug] hasUnrenderedMath called (attempt=${attempts}) → ${unrendered}`
      );
      return unrendered;
    }

    /* ---- 3c. Core render function ---- */
    function run() {
      console.log("[MathJax Debug] ▶ Running MathJax render sequence...");

      // --- Wait until MathJax object exists ---
      if (
        !window.MathJax ||
        !window.MathJax.startup ||
        !window.MathJax.typesetPromise
      ) {
        console.log("[MathJax Debug] MathJax not yet ready — waiting...");
        return setTimeout(run, 300);
      }

      // --- Wait for both MathJax startup and font loading ---
      Promise.all([
        window.MathJax.startup.promise,
        document.fonts ? document.fonts.ready : Promise.resolve(),
      ])
        .then(() => {
          console.log(
            "[MathJax Debug] ✅ Fonts + MathJax startup complete, preparing typeset..."
          );

          /* --------------------------------------------------------------
           * FIX #1: wait one frame so CSSOM is guaranteed live
           * FIX #2: then poll until every MathJax style sheet (data-mjx-style)
           *         is attached and populated with cssRules.
           * ------------------------------------------------------------ */
          return new Promise((resolve) => {
            requestAnimationFrame(() =>
              setTimeout(() => {
                console.log(
                  "[MathJax Debug] Frame delay passed → typesetClear()"
                );
                window.MathJax.typesetClear();

                // ---- Additional safeguard: wait until stylesheets are ready ----
                const waitForStyleSheets = () => {
                  const sheets = Array.from(document.styleSheets).filter(
                    (s) => s.ownerNode && s.ownerNode.dataset?.mjxStyle
                  );
                  const allReady = sheets.every((s) => {
                    try {
                      return !!s.cssRules.length;
                    } catch {
                      return false;
                    }
                  });
                  if (allReady) {
                    console.log(
                      "[MathJax Debug] ✅ All mjx stylesheets attached (" +
                        sheets.length +
                        ")"
                    );
                    resolve();
                  } else {
                    console.log(
                      "[MathJax Debug] ⏳ Waiting for mjx stylesheets to attach..."
                    );
                    requestAnimationFrame(waitForStyleSheets);
                  }
                };
                waitForStyleSheets();
              }, 50)
            );
          });
        })
        .then(() => {
          console.log("[MathJax Debug] Calling MathJax.typesetPromise()…");
          return window.MathJax.typesetPromise([root]);
        })
        .then(() => {
          /* --- Post-typeset cleanup & verification --- */
          document.querySelectorAll(".mjx-container svg").forEach((svg) => {
            svg.style.overflow = "visible"; // prevent clipping
          });

          if (hasUnrenderedMath(root) && attempts < 2) {
            attempts++;
            console.warn(
              `[MathJax Debug] ⚠️ Unrendered math still found (attempt=${attempts}) → retrying…`
            );
            setTimeout(run, 400 * attempts);
          } else {
            console.log("[MathJax Debug] ✅ MathJax rendering complete.");
          }
        })
        .catch((e) =>
          console.error("[MathJax Debug] ❌ MathJax typeset error:", e)
        );
    }

    /* ---- Kick off initial render ---- */
    run();
  });
})();
