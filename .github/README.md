# Responsible AI Agent Teams for Actuarial Risk Prediction
## MSc AI Thesis Proposal

## 1. Introduction

Predictive modelling techniques have been broadly accepted in the insurance industry for many years, where risk prediction and differentiation are essential for maintaining profitability. Since insurers do not know in advance what their actual claim costs will be in a given year, they rely on predictive models to estimate claim and fraud risk, for example. Using these estimates to appropriately underwrite business and set insurance premiums ensures solvency and profitability. Without accurate predictions, insurers risk either undercharging policies, leading to financial losses, or overcharging them, which may drive customers away.

Consequently, insurers continuously seek to increase data insights and enhance predictive models to obtain more accurate predictions. Insurers traditionally relied on univariate analysis and Generalized Linear Models (GLMs), but moved towards machine learning techniques to enhance predictive performance. Still, most modelling projects require intensive actuarial resources and lack adaptability and automation.

In this context, generative agents, autonomous, reasoning‑capable AI entities powered by large language models (LLMs), present both an opportunity and a challenge. Such agents can dynamically adjust risk classification based on evolving data and criteria. However, their autonomy demands new forms of explainability and oversight.

This thesis explores how a team of generative agents can perform actuarial risk classification, with a focus on explainability (XAI), fairness, and accountability. The thesis investigates whether collaborative and specialized AI agents can deliver transparent and auditable classifications that align with actuarial and regulatory standards. As reasoning and interpretation are key aspects of the actuarial profession, generative agents are employed in this thesis because of their reasoning and natural language explanation capabilities.

## 2. Project Description

While generative AI agents can autonomously optimize risk modelling, they may produce decisions that are difficult to explain, audit, or justify—especially when using non‑linear or opaque models. For actuaries who are considering employing generative agents for actuarial risk classification, the following aspects are crucial:

- **Explainability & transparency**: Modelling decisions must be understandable; predictions for specific groups or policyholders need to be interpretable.
- **Consistency & robustness**: As risk classification strongly drives insurance premiums, predictions need a certain degree of stability, preventing too much randomness and volatility across different model runs.
- **Accuracy**: Model performance is naturally an important indicator for actuaries.

### Research Questions

1. Can a team of generative AI agents, each with specialized roles, be designed to autonomously perform actuarial risk classification, while adhering to performance and interpretability constraints?
2. Can generative AI agents for risk classification be deployed in a responsible and explainable way? How could XAI techniques be applied and operationalized for this specific problem? How effective are traditional XAI methods (e.g., SHAP, LIME) when applied to collaborative agent‑based models?
3. Can a setup with generative agents lead to consistent and robust outcomes? How can this be integrated in the setup?

### Theoretical Framework

This thesis is grounded in three interconnected areas of research:

- **Actuarial science and predictive modelling**: Traditional risk classification in insurance is based on statistical models such as Generalized Linear Models (GLMs), which emphasize interpretability and regulatory compliance. Recent advances in machine learning provide improvements in predictive accuracy but often lack transparency and adaptability.
- **Multi‑agent systems (MAS) and LLM‑based agents**: Building on research in distributed AI, generative agents are autonomous, reasoning‑capable entities powered by large language models (LLMs). In this thesis, MAS principles are applied to structure a team of specialized agents (e.g., data preparation, modelling, auditing, explanation validation), each contributing to risk classification tasks.
- **Explainable and Responsible AI (XAI)**: As actuarial decisions must be auditable and accountable, the work intends to apply explainability frameworks for evaluating robustness, consistency, and fairness. These provide a basis for assessing whether AI agent teams can meet actuarial and regulatory standards.

### Contributions to Existing Research

- A working prototype of a generative AI agent team for actuarial risk classification.
- A novel application of LLMs in predictive, explanatory, and reflective insurance tasks.
- A modular and replicable framework for integrating performance and actuarial XAI needs into multi‑agent systems.

### Architecture

<figure align="center">
  <img src="C:/Users/bart_/Documents/Thesis/project_management/thesis_architecture_v3_no_background.png" alt="High-level architecture of the agent team" width="600">
  <figcaption><strong>Figure 1.</strong> High-level architecture of the generative AI agent team for actuarial risk classification.</figcaption>
</figure>
