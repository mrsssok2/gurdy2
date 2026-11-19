---
title: Gurdy 2.0 demo
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Simulated six-legged robot mission control (no real robot)
---

# Gurdy 2.0 - mission control demo

A **simulated** hexapod in a two-room lab. Type commands such as "go to room B", "find the yellow box", "where are you?", "go home" and press E-STOP.
Nothing here controls a real robot. Every visitor gets a private simulated robot (freed after 10 idle minutes; if the demo is full, try again in a minute) and the planner is a small keyword parser (the real project uses a fine-tuned local LLM, voice and Gazebo).

Source: the Gurdy 2.0 project (README with the full system and results).
