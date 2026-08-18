# E-Commerce KT RAG System

This project converts the **E-Commerce Order Management System KT PDF** into a simple Retrieval-Augmented Generation (RAG) application.

## Architecture

User Question
    |
    v
Streamlit UI
    |
    v
Question Embedding
    |
    v
FAISS Vector Search
    |
    v
Top-K KT Chunks
    |
    v
OpenAI Responses API
    |
    v
Grounded Answer + Retrieved Pages

## Knowledge Base

The application uses:

**Knowledge Transfer Document – E-Commerce Order Management System.pdf**

The KT covers:
- Project introduction and business requirements
- Layered architecture
- React, Java/Spring Boot, MySQL, REST APIs and Git
- Development, QA, UAT and Production environments
- Registration and login
- Customer profile
- Product search and product details
- Shopping cart and inventory
- Checkout and order creation
- Order lifecycle
- Payment processing and payment failure handling
- Order and order-item database structure
- REST APIs
- Controller, service and repository layers
- Exception handling and logging
- Security
- Git workflow and code review
- Testing
- CI/CD
- Production deployment and support
- Monitoring and incident management
- Knowledge required for a new developer

## Setup

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Rename `.env.example` to `.env`.
5. Add your OpenAI API key to `.env`.
6. Keep the KT PDF in the same folder as `app.py`.
7. Start the application:

```bash
streamlit run app.py
```

8. Open the local Streamlit URL shown in the terminal.

## Example questions

- What is the architecture of the application?
- What is the order lifecycle?
- What happens during checkout?
- What happens if payment times out?
- What are the main REST APIs?
- What is the responsibility of the service layer?
- What information is stored in ORDER_ITEM?
- What is the Git workflow?
- How is the application tested?
- What should a developer check when an order is pending after successful payment?

## Why this is RAG

The system does not simply ask the LLM to answer from general knowledge.

It first:
1. extracts the KT document,
2. creates searchable embeddings,
3. retrieves the most relevant chunks for the question,
4. places those chunks into the LLM context,
5. generates an answer restricted to the retrieved KT content.

This reduces unsupported answers and makes the response traceable to the KT pages.
