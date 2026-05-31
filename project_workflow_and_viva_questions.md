# Vision-Augmented LLM Pipeline for Long-Form Document Summarization
## Architecture Overview & Viva Voce Guide

### 1. Abstract
This project is an advanced contextual document summarizer. Unlike standard RAG systems that only parse raw string text, this pipeline utilizes a Vision-Augmented approach to "read" long-form documents (like research papers) visually. It uses deep learning models to detect bounding boxes for abstract, introduction, tables, figures, and equations, extracting them accurately without garbling the formatting.

### 2. Core Workflow
1. **Input Ingestion**: The user uploads a PDF, provides a URL, pastes a YouTube link, or uploads an image.
2. **Vision Extraction (PDFs)**:
    - **Rendering**: PyMuPDF renders the PDF pages into high-res images.
    - **Layout Detection**: LayoutParser (PubLayNet) predicts bounding boxes for Text, Titles, Tables, Figures, and Lists.
    - **Modality Handling**: 
        - Text is cropped using `pdfplumber`.
        - Tables are extracted using `img2table` (fallback to `pdfplumber`).
        - Images are captioned using BLIP.
        - Equations are transcribed using `pix2tex` (LaTeX OCR).
3. **Semantic Assembly**: The document is reconstructed in reading order. For research papers, the Abstract, Introduction, and Conclusion are identified.
4. **Summarization**: The text is sent to HuggingFace Transformers (e.g., BART, Pegasus). If it exceeds the context window, recursive chunked summarization is applied.
5. **Evaluation**: Summaries are evaluated using ROUGE (1, 2, L) and BLEU metrics.

### 3. Viva Voce Questions

**Q1: Why did you use LayoutParser instead of just PyPDF2 or PyMuPDF for text extraction?**
**Answer:** Standard parsers extract text blindly and linearly. In two-column research papers, this causes text from the left column to be concatenated with the right column, resulting in gibberish. LayoutParser uses a Faster R-CNN model to visually identifying column boundaries, ensuring we extract text logically in reading order.

**Q2: How do you handle complex elements like Tables or Math Equations?**
**Answer:** Tables are often destroyed by standard text scrapers. We isolate the table bounding box and pass it to `img2table` to generate a structured Markdown table. For math equations, we use `pix2tex` to perform Optical Character Recognition specifically tuned for LaTeX sequences.

**Q3: What happens if the HuggingFace Summarizer models cannot handle the length of the document?**
**Answer:** The maximum context size for models like BART is 1024 tokens. We use recursive chunking: the document is split into overlapping chunks, summarized individually, and then the intermediate summaries are aggregated and summarized once more to form the final output.

**Q4: How do you measure the quality of your summaries?**
**Answer:** We compute the ROUGE score and BLEU score. ROUGE focuses on recall (how many n-grams from the reference are in the generated summary), while BLEU focuses on precision. We also analyze the abstractiveness to distinguish between extractive and abstractive models.

**Q5: Can you explain how the BLIP model fits into text summarization?**
**Answer:** BLIP is a Vision-Language foundation model. When LayoutParser detects a figure or diagram in a document, we pass the image crop to BLIP to generate a textual description of the image. This description is injected into the document text as `[FIGURE: description]`, allowing the LLM summarizer to "read" the images.
