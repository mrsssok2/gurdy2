---
title: Gurdy 2.0 demo
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Simulated six-legged robot mission control, runs in your browser
---

# Gurdy 2.0 - mission control demo

A **simulated** hexapod in a two-room lab. Type commands such as "go to room B", "find the yellow box", "where are you?", "go home" and press E-STOP.
Everything runs in your browser (the real Python simulation and agent, via Pyodide): no server, nothing is sent anywhere, every visitor has a private robot, and a reload starts over.
The planner is a small keyword parser (the real project uses a fine-tuned local LLM, voice and Gazebo). The first visit downloads about 6 MB.
