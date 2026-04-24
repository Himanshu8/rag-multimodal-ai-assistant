# Multimodal AI Knowledge Assistant

A Streamlit-based application that integrates image understanding, document retrieval (RAG), and language model reasoning into a unified system for interactive knowledge extraction.

---

## Overview

This project combines computer vision and natural language processing to enable users to:
- Generate captions from images  
- Ask questions from uploaded documents  
- Receive context-aware answers using a retrieval-augmented generation (RAG) pipeline  

The system is modular, efficient, and deployed using Streamlit for interactive use.

---

## Features

### Image Captioning
- Generates descriptive captions from input images  
- Uses a pretrained BLIP model for vision-language understanding  
- Supports JPG, PNG, and JPEG formats  

### Document Question Answering (RAG)
- Supports PDF, TXT, and DOCX files  
- Extracts and processes document text  
- Splits text into overlapping chunks for better context retention  
- Builds a FAISS index for similarity-based retrieval  
- Retrieves relevant content based on user queries  

### Language Model Reasoning
- Uses an instruction-tuned FLAN-T5 model  
- Generates concise and context-aware answers  
- Optimized for efficient inference  

---

## System Workflow

### Image Pipeline
1. User uploads an image  
2. Image is processed using the BLIP processor  
3. Caption is generated using the pretrained model  

### Document RAG Pipeline
1. User uploads a document (PDF, TXT, DOCX)  
2. Text is extracted from the document  
3. Text is split into overlapping chunks  
4. Embeddings are generated using Sentence Transformers  
5. FAISS index is created for similarity search  
6. Relevant chunks are retrieved based on the query  
7. Retrieved context is passed to the language model  
8. Final answer is generated and displayed  

---

## Technologies Used

- Streamlit (User Interface)  
- PyTorch  
- Transformers (Hugging Face)  
- BLIP (Salesforce) for image captioning  
- FLAN-T5 (Google) for text generation  
- Sentence Transformers (MiniLM) for embeddings  
- FAISS for vector similarity search  
- PyPDF for PDF parsing  
- python-docx for DOCX parsing  
- Pillow for image processing
