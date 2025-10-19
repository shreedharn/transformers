(function () {
  const onPageChange = (fn) => {
    if (window.document$ && window.document$.subscribe) {
      console.log("[MathJax Debug] Subscribed to document$ navigation events");
      window.document$.subscribe(({ body }) => {
        console.log("[MathJax Debug] New page loaded via instant navigation");
        fn(body);
      });
    } else {
      console.log("[MathJax Debug] Using fallback on window.load");
      window.addEventListener('load', () => fn(document.body), { once: true });
    }
  };

  onPageChange((root) => {
    let attempts = 0;

    function hasUnrenderedMath(el) {
      const txt = el.innerText || '';
      const unrendered = /\$\$|\\begin\{/.test(txt);
      console.log(`[MathJax Debug] hasUnrenderedMath called (attempt=${attempts}) → ${unrendered}`);
      return unrendered;
    }

    function run() {
      console.log("[MathJax Debug] Running MathJax render sequence...");
      if (!window.MathJax || !window.MathJax.startup) {
        console.log("[MathJax Debug] MathJax not yet ready — waiting...");
        return setTimeout(run, 150);
      }

      window.MathJax.startup.promise
        .then(() => {
          console.log("[MathJax Debug] Calling MathJax.typesetPromise()");
          return window.MathJax.typesetPromise([root]);
        })
        .then(() => {
          if (hasUnrenderedMath(root) && attempts < 1) {
            attempts++;
            console.warn("[MathJax Debug] Unrendered math found — retrying once...");
            setTimeout(() => window.MathJax.typesetPromise([root]), 150);
          } else {
            console.log("[MathJax Debug] MathJax rendering complete.");
          }
        })
        .catch(e => console.error("[MathJax Debug] MathJax typeset error:", e));
    }

    run();
  });
})();
