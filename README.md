# Modular Local AI Assistant

## Project Overview

This project aims to build a fully local, modular, and GPU-accelerated AI assistant designed to run offline on a Linux-based mini PC with an eGPU. Its core functionality includes Retrieval Augmented Generation (RAG) from local documents, interaction with a local Large Language Model (LLM), and persistent conversational memory, all operating within Docker containers. A Streamlit web-based Graphical User Interface (GUI) is provided for interactive chat.

## Current Features

* **Local LLM Integration:** Uses LLAMA 3 (via Ollama) for text generation.
* **Document Question Answering (RAG):** Answers questions by retrieving relevant information from local documents (PDFs, TXT, CSV, MD, HTML, DOCX) using LangChain and FAISS.
* **Configurable RAG Relevance:** Features a GUI slider to dynamically adjust the similarity threshold for document retrieval, allowing fine-tuning of context inclusion.
* **Persistent Conversational Memory:** Stores chat history to a local JSON file on an external SSD, maintaining context across sessions.
* **GPU Acceleration:** Leverages an NVIDIA RTX 3060 eGPU for efficient embeddings and LLM inference.
* **Containerized Environment:** All core components run within Docker containers for isolated and reproducible setup.
* **Streamlit Web GUI:** Provides a user-friendly chat interface accessible via a web browser.

## Hardware Requirements

* **Linux-based Mini PC:** (e.g., Minisforum UM760 with AMD Ryzen 57640HS CPU, 32 GB RAM, 1TB NVMe internal storage).
* **External GPU (eGPU):** NVIDIA RTX 3060 (12 GB VRAM) inside an eGPU enclosure (e.g., Razer Core).
* **External SSD:** 1 TB PNY SSD (recommended for model/data storage).
* **Internet Connection (for initial setup):** Required for cloning repo, downloading Docker images, installing packages, and pulling Ollama models.

## Software Requirements (Host System - Ubuntu 24.04 Recommended)

* **Operating System:** Ubuntu 24.04 (Noble Numbat) or similar Linux distribution.
* **Docker & Docker Compose:** Latest stable versions.
* **NVIDIA Drivers:** Compatible with your RTX 3060 (e.g., version 570.133.07 used in development).
* **NVIDIA Container Toolkit:** For Docker to access the GPU.
* **Git:** For cloning the repository.
* **Ollama:** Installed directly on the host system to serve LLMs.

## Setup & Installation Guide

Follow these steps carefully to get the AI assistant up and running.

### 1. Clone the Repository

Open a terminal on your Linux host machine and clone the project:

```bash
git clone [https://github.com/Bwenz68/modular-local-ai-assistant.git](https://github.com/Bwenz68/modular-local-ai-assistant.git)
cd modular-local-ai-assistant
