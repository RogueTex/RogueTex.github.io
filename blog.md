---
layout: default
title: "Blog"
permalink: /blog
---

<div class="container mx-auto px-4 py-8">
  <h1 class="text-3xl font-semibold mb-6">All Blog Posts</h1>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {% for post in site.posts %}
    <a href="{{ post.url }}" class="block bg-white rounded-lg shadow hover:shadow-md transition">
      <img class="w-full h-40 object-cover rounded-t-lg" src="/assets/images/post-placeholder.jpg" alt="{{ post.title }}">
      <div class="p-4">
        <h2 class="text-xl font-bold">{{ post.title }}</h2>
        <p class="text-gray-600 mt-2">{{ post.excerpt | strip_html | truncatewords: 24 }}</p>
        <p class="text-sm text-gray-500 mt-2">{{ post.date | date: "%B %d, %Y" }}</p>
      </div>
    </a>
    {% endfor %}
  </div>
</div>
