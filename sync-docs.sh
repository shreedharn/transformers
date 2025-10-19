rm -rf site/

mkdocs build

rsync -a --include='*.md' \
         --include='styles.css' \
         --exclude='*' \
         ./ docs/
        
rsync -a \
  --include='javascripts/' \
  --include='javascripts/mathjax-init.js' \
  --include='javascripts/mathjax-refresh.js' \
  --exclude='*' \
  ./ docs/