window.MathJax = {
  tex: {
    packages: {'[+]': ['ams', 'textmacros']},
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  chtml: {
    displayAlign: "left",
    displayIndent: "0",
    linebreaks: { automatic: true, width: "container" },
    adaptiveCSS: false,
  },
  svg: {
    displayAlign: "left",
    displayIndent: "0",
    linebreaks: { automatic: true, width: "container" },
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  },
  startup: {
    typeset: false 
  }
};
