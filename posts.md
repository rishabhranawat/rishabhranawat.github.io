---
layout: page
title: Posts
---

<p class="message">
	An index of everything I've written here — notes on papers, technical projects, and open-source work.
</p>

{% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
<h2>{{ year.name }}</h2>
<ul class="post-index">
  {% for post in year.items %}
  <li>
    <span class="post-index-date">{{ post.date | date: "%b %-d" }}</span>
    <a href="{{ post.url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>
{% endfor %}

<style>
.post-index {
  list-style: none;
  padding-left: 0;
}
.post-index li {
  margin-bottom: 0.4rem;
}
.post-index-date {
  display: inline-block;
  min-width: 4.5rem;
  color: #9a9a9a;
  font-size: 0.85em;
}
</style>
