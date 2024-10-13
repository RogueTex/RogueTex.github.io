---
layout: default
title: "Blog"
---

<div class="container">
  <h1>All Blog Posts</h1>
  <div class="row">
    {% for post in site.posts %}
    <div class="col-lg-4 mb-4">
      <div class="card">
        <img class="card-img-top" src="/assets/images/post-placeholder.jpg" alt="{{ post.title }}">
        <div class="card-body">
          <h5 class="card-title">{{ post.title }}</h5>
          <p class="card-text">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
          <a href="{{ post.url }}" class="btn btn-primary">Read More</a>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</div>

