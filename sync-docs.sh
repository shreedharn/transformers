rsync -a --include='*.md' \
         --include='styles.css' \
         --exclude='*' \
         ./ docs/
        
rsync -a \
  --include='javascripts/' \
  --include='javascripts/mathjax.js' \
  --exclude='*' \
  ./ docs/