#!/bin/bash
./sync-docs.sh
mkdocs build
mkdocs gh-deploy -b gh-pages --force