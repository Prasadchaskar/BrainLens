# BrainLens

Brain MRI tumor classification and MLOps platform.

## Objective

BrainLens is an end-to-end MLOps application for brain MRI tumor
classification.

## Current Model

EfficientNet-B0 transfer learning.

Classes:

- Glioma
- Meningioma
- No Tumor

Current baseline:

- Validation Accuracy: 96.61%
- Test Accuracy: 92.55%

## Architecture

BrainLens follows a service-based MLOps architecture.

Services:

- Data Download Service
- Data Validation Service
- Preprocessing Service
- Training Service
- Evaluation Service
- Model Registry Service
- Prediction Service
- Drift Monitoring Service

Infrastructure:

- Airflow
- Docker
- Kubernetes