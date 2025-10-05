window.MathJax = {
  tex: {
    packages: {'[+]': ['ams', 'textmacros']},  
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  chtml: {
    displayAlign: "left",          // left-align equations
    displayIndent: "0",            // optional: indent amount
    linebreaks: { automatic: true, width: "container" }
  },
  svg: {
    displayAlign: "left",          // left-align equations
    displayIndent: "0",
    linebreaks: { automatic: true, width: "container" }
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};