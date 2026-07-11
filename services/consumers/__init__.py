"""Kafka consumers for event-driven workflows.

Currently hosts the real-time risk-recomputation consumer (Tier 7), which
re-scores a patient's hereditary risk when a new clinical or family-graph event
arrives and raises a notification if the risk profile changes materially.
"""
