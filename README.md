# 🌍 Climate Action Predictor – AI for SDG 13

![Python](https://img.shields.io/badge/Python-3.8%252B-blue) ![ML](https://img.shields.io/badge/Machine%2520Learning-Scikit--learn-orange) ![UN SDG 13](https://img.shields.io/badge/UN%2520SDG-13%2520Climate%2520Action-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## **1. Project Overview**
This machine learning project addresses **UN Sustainable Development Goal 13: Climate Action**.  
It predicts CO₂ emissions and evaluates climate mitigation strategies to support **data-driven policymaking** and sustainable development.

**Objectives:**
- Predict per capita CO₂ emissions based on economic and environmental indicators  
- Quantify effectiveness of climate strategies  
- Prioritize high-impact interventions  
- Simulate complex policy interactions  

---

## **2. Problem Statement**
Climate change is a critical global challenge. Policymakers face:

- Limited predictive tools for emissions forecasting  
- Lack of quantitative strategy analysis  
- Difficulty prioritizing interventions  
- Challenges simulating multi-policy interactions  

**Goal:** Enable informed and efficient climate action using AI.

---

## **3. Machine Learning Solution**
**Model:** Random Forest Regressor (200 trees)  
**Features:** 7 economic & environmental indicators  

**Functions:**
- Predicts CO₂ emissions per capita  
- Identifies drivers via feature importance  
- Simulates strategy impact  
- Provides data-driven policy recommendations  

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| R² Score | 0.9920 |
| Mean Absolute Error | 162.39 tons/capita |

**Feature Importance:**

| Feature | Importance | Insight |
|---------|------------|---------|
| GDP Per Capita | 92.7% | Economic growth dominates emissions |
| Energy Consumption | 4.2% | Energy intensity of economy |
| Vehicle Density | 1.1% | Transport sector emissions |
| Industrial Output | 0.7% | Manufacturing impact |
| Urbanization | 0.6% | Urban development patterns |
| Renewable Energy | 0.4% | Adoption of clean energy |
| Forest Area | 0.3% | Natural carbon sequestration |

---

## **4. Climate Strategy Impact Assessment**
**Sample Country Profile:**

- GDP/capita: $15,000  
- Renewable Energy: 20%  
- Industrial Output: 25% of GDP  
- Forest Coverage: 35%  
- Vehicle Density: 250/1000 people  
- Energy Consumption: 2,500 kWh/capita  
- Urbanization: 60%  
- Baseline Emissions: 3,279.3 tons CO₂/capita  

**Strategy Comparison:**

| Strategy | New Emissions | Reduction | % Reduction | Key Actions |
|----------|---------------|-----------|-------------|------------|
| Comprehensive Green Deal | 3,185.0 | 94.3 | 2.9% | Multi-sector approach |
| Renewable Energy Transition | 3,231.6 | 47.7 | 1.5% | +30% renewables, -10% energy use |
| Sustainable Transportation | 3,236.4 | 43.0 | 1.3% | -25% vehicles, +10% urbanization |
| Industrial Decarbonization | 3,250.5 | 28.9 | 0.9% | -5% industry, +20% renewables |
| Forest Conservation | 3,281.7 | -2.4 | -0.1% | +20% forest, -8% industry |

**Key Insights:**
- **Economic Transformation:** Decouple GDP growth from emissions  
- **Comprehensive Policy Packages:** Multi-sector approach is most effective  
- **Energy Efficiency:** Reduce 4.2% emissions from energy use  
- **Sustainable Mobility:** Reduce 1.1% emissions from transport  

---

## **5. Technical Architecture**

```mermaid
graph TD
A[Economic & Environmental Data] --> B[Data Preprocessing]
B --> C[Feature Engineering]
C --> D[Random Forest Model]
D --> E[Emission Predictions]
D --> F[Feature Importance Analysis]
E --> G[Strategy Simulation Engine]
F --> H[Policy Impact Assessment]
G --> H
H --> I[Evidence-Based Recommendations]
