# Streamlit Navigation Guide

This document explains the purpose of each page in the Streamlit sidebar navigation for the Healthcare Hereditary Disease Prediction System.

## Navigate

The sidebar `Navigate` control is the main entry point into the app. It lets users switch between operational views for patient management, screening, prediction, analytics, genetics, patient-facing tools, and compliance monitoring.

## Pages

### 📊 Dashboard

High-level home page for the system. Use this page to review the core KPIs, see the current risk-score distribution, inspect summary trends by demographics, and view recent prediction activity.

### 👤 Patient Management

Clinical CRUD workspace for patient records. Use it to register and update patients, manage conditions, track family members, and maintain medication lists.

### 🏥 Encounters & Vitals

Operational page for visit tracking. Use it to start and manage patient encounters, record vitals during a visit, and close encounters when the visit ends.

### 📊 Batch Screening

Cohort screening page for panel-wide predictions. Use it to run risk screening across groups of patients and review longitudinal prediction trends over time.

### 🔮 Risk Prediction

Patient-level scoring page. Use it to enter demographics, medical history, family-history factors, and comorbidities, then generate an individual hereditary disease risk score and recommendation band.

### 👨‍👩‍👧 Family Tree

Pedigree visualization page. Use it to inspect family relationships, identify affected relatives, and understand hereditary risk patterns across generations.

### 🤖 Model Training

Model development and evaluation page. Use it to train the predictive model, compare performance metrics, inspect ROC and confusion matrix outputs, and review feature importance.

### 📈 Analytics

System monitoring page. Use it to track data quality, service health, prediction volume over time, and simple model-drift signals.

### 🧬 Genetics

Genetics and genomics tooling page. Use it for Mendelian inheritance calculations, cascade screening previews, variant annotation, and polygenic risk scoring.

### 🧠 Decision Support

Clinical decision-support workspace. Use it to simulate what-if changes to a patient profile, review drift and fairness checks, generate guideline-based recommendations, and suggest missing pedigree links.

### 🌍 Population Health

Panel-level population analytics page. Use it to review aggregate screening coverage, average risk by demographic group, and overall risk distribution across the patient panel.

### 🔔 Notifications

Alerting page for risk changes. Use it to review threshold-based and rising-risk notifications derived from patient prediction history.

### 🔐 Patient Portal & Consent

Patient-facing self-service page. Use it to record consent decisions, review effective consent state, and preview the lay-friendly portal view with de-identified family information.

### 🔒 Audit Logs

Admin-only compliance page. Use it to review who accessed or changed resources, filter events by actor or action, and export audit activity for review.

## Summary

Together, these pages cover the full workflow of the application: data capture, encounter tracking, batch screening, individual prediction, genetics, decision support, population analytics, notifications, patient self-service, and audit/compliance oversight.