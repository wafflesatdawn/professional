---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
format: hugo-md
  jupyter: python3
  draft: true
execute: 
  freeze: true
---