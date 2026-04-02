(function() {
  const script = document.createElement("script");
  script.src = "https://unpkg.com/mermaid@10.2.4/dist/mermaid.min.js";
  script.onload = () => {
    if (window.mermaid) {
      mermaid.initialize({ startOnLoad: true });
      console.log("✅ Mermaid initialized (classic).");
    } else {
      console.error("❌ Mermaid failed to load.");
    }
  };
  document.head.appendChild(script);
})();
