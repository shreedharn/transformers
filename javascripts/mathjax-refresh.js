// Re-run MathJax every time MkDocs loads new content
document$.subscribe(() => {
  if (window.MathJax) {
    window.MathJax.typesetPromise();
  }
});
