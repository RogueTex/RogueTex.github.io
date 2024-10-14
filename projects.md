---
layout: page
title: "Projects"
permalink: /projects/
---

## My Projects

Here are some of my recent projects:

<ul>
  {% for project in site.projects %}
    <li>
      <h3>{{ project.title }}</h3>
      <p><strong>Date:</strong> {{ project.date | date: "%B %d, %Y" }}</p>
      <p><strong>Brief:</strong> {{ project.description }}</p>
    </li>
  {% endfor %}
</ul>

