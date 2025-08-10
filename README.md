---
tags:
- mcp-server-track
- agent-demo-track
- consilium
- Mistral
- SambaNova
title: Consilium MCP Server
emoji: 🏢
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: true
short_description: Multi-AI Expert Consensus Platform
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/65ba36f30659ad04d028d083/Ubq8rnD8WhWpw8Qt32_lk.png
---

## Videos

* 📼 Video UI: <https://youtu.be/ciYLqI-Nawc>
* 📼 Video MCP: <https://youtu.be/r92vFUXNg74>

## Features

* Visual roundtable of the AI models, including speech bubbles to see the discussion in real time.
* MCP mode enabled to also use it directly in, for example, Claude Desktop (without the visual table).
* Includes Mistral (**mistral-large-latest**) via their API and the Models **DeepSeek-R1**, **Meta-Llama-3.3-70B-Instruct** and **QwQ-32B** via the SambaNova API.
* Research Agent with 5 sources (**Web Search**, **Wikipedia**, **arXiv**, **GitHub**, **SEC EDGAR**) for comprehensive live research.
* Assign different roles to the models, the protocol they should follow, and decide the communication strategy.
* Pick one model as the lead analyst (had the best results when picking Mistral).
* Configure the amount of discussion rounds.
* After the discussion, the whole conversation and a final answer will be presented.
