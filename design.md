# Application Design

**Topology:** Star topology (central hub with logging and memory).  
**Agents:** Data Prep, Modelling, Auditing, Explanation Validation.  
**Collaboration Type:** Cooperative (shared goal).  
**Strategy:** Layered reasoning (structured steps) + Belief Revision (feedback loop).  

**Hub responsibilities:**
- Route messages between agents
- Store audit logs (JSON format)
- Track belief changes (before vs after revision)
- Provide traceability for regulatory compliance

**Common message and memory schema**
{
  "task": "train_model",
  "payload": {
    "data_id": "train_set_v1",
    "parameters": {"model_type": "PoissonGLM"}
  },
  "metadata": {
    "sender": "Data Prep Agent",
    "version": "0.1",
    "timestamp": "auto"
  }
}

### Prompt Template Examples

**Modelling Agent – System Prompt**
> You are an actuarial modelling assistant.  
> Given structured training data, fit an interpretable model (GLM/Poisson)  
> and output predictions and key feature influences in JSON.

**Explanation Agent – System Prompt**
> You validate and generate natural language explanations  
> based on SHAP/LIME outputs. Ensure explanations align with actuarial logic.