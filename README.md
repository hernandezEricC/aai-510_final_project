# Review-to-Revenue: AI Assistant for Small Businesses

This project is part of the **AAI-510-02 course** in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: Completed


## Installation

To use this project, first clone the repo on your device using the command below:
```
git init
git clone https://github.com/hernandezEricC/aai-510_final_project
```

## Project Intro / Objective
The objective of this project is to build an agent that analyzes Google reviews and business metadata to provide decision-support and intelligence for local business owners. The agent identifies customer complaints, compares businesses against nearby competitors, identifies opportunities based on low and high reviews, and suggests operational changes to improve customer satisfaction.

To demonstrate proof of concept, this project focuses only on Italian restaurants in San Diego, CA. In future expansions, the data pipe line can be expanded to include any business type and location.


## Contributors
- Brandon Wirgau
- Eric Hernandez
- Erika Gallegos


## Methods Used
- Data Cleaning & Preparation
- Agent Building
- Agent Validation
- LLM Judge

## Technologies
- Python  
- Databricks
- Unity Catalog
- GTE Large (En)
- LangGraph
- Meta Llama 3.3 70B
- OpenAI GPT OSS 60B


## Project Description
This project develops an AI-powered restaurant comparison agent designed to help small business owners transform customer reviews into actionable business insights. Using Google Maps review data and business metadata from Italian restaurants in San Diego, California, the agent enables users to compare restaurants, identify strengths and weaknesses, and better understand customer perceptions. Rather than requiring users to manually analyze thousands of reviews, the agent synthesizes both quantitative metrics and qualitative customer feedback into concise, evidence-based responses.

The agent combines structured business information, such as ratings and review counts, with semantic retrieval of customer review text. Through specialized tools, users can compare competing restaurants, explore customer sentiment and recurring themes, and uncover factors that contribute to positive or negative dining experiences. Guardrails ensure that responses remain grounded in the available data, prevent hallucinated restaurant information, and gracefully reject requests outside the agent's intended scope.

The final solution demonstrates how domain-specific AI agents can support data-driven decision making by distilling large volumes of unstructured customer feedback into digestible insights. While the project was developed using a focused subset of San Diego Italian restaurant data to maintain efficient vector search performance, the architecture can be extended to additional business categories and larger datasets in future implementations.

The project is comprised of two notebooks, one for the data pipeline (data_pipeline.ipynb) and one for the agent definition and evaluation (tool_definitions.ipynb).

## Acknowledgements
The authors want to acknowledge the original authors of the dataset, which can be found at: https://mcauleylab.ucsd.edu:8443/public_datasets/gdrive/googlelocal/.
